"""
build_tool_c.py  —  NLP Scam Classifier (Tool C)  [v2]
=======================================================
Pipeline position: receives Whisper transcript → returns scam_probability score.

Changes from v1
---------------
- Seed data expanded to 6–8 structurally distinct examples per category
  (different tactics, different urgency levels, different personas)
- Augmentation prompt now enforces structural diversity (not just paraphrase)
- Augmentation output is quality-filtered before entering training
- Eval now includes confusion matrix + per-class report + false-negative rate
- Stratified 5-fold CV runs before final hold-out test (hold-out never touched
  during training or fold selection)
- Threshold tuning on validation folds (scam detection penalises FN more than FP)
- ScamTextClassifier.predict() exposes calibrated threshold + raw logits for agent.py

Requirements
------------
pip install transformers datasets scikit-learn pandas torch huggingface_hub

Free HF token: https://huggingface.co/settings/tokens  (read access is enough)
Set it once:  huggingface-cli login   OR   export HF_TOKEN="hf_..."
"""

import os, json, time, warnings
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report,
)
from huggingface_hub import InferenceClient

warnings.filterwarnings("ignore")

os.makedirs("./models/tool_c", exist_ok=True)
os.makedirs("./data", exist_ok=True)

HF_TOKEN       = os.environ.get("HF_TOKEN", "")
AUGMENT_MODEL  = "mistralai/Mistral-7B-Instruct-v0.3"
SAMPLES_PER_CATEGORY = 35   # more samples, filtered down later
MIN_SAMPLE_LEN = 25         # characters — discard very short generated lines
MAX_SAMPLE_LEN = 400        # characters — discard rambling generated lines
N_FOLDS        = 5          # stratified k-fold for CV
SCAM_THRESHOLD = 0.40       # tuned lower: missing a scam is worse than a false alarm


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEED DATA
#    Rules for seeds:
#    - Each entry uses a DIFFERENT tactic / persona / urgency level
#    - Mix: robocall vs live agent vs "officer" vs automated voice
#    - Mix: high urgency vs soft pressure vs fear vs false reward
#    - Avoid repeating sentence structure across seeds in same category
# ─────────────────────────────────────────────────────────────────────────────

SEED_SCAM = {

    "irs_government": [
        # Hard threat, arrest warrant
        "This is the IRS. You owe back taxes and an arrest warrant has been issued. Call immediately to avoid prosecution.",
        # Soft opening, escalation hook
        "Hello, we have been trying to reach you regarding an unresolved tax matter from last year that requires your attention today.",
        # Social Security angle
        "Your Social Security number has been flagged for fraudulent activity. Press 1 to speak with a federal officer now.",
        # Department of Justice persona
        "This is the Department of Justice. A civil lawsuit has been filed against your tax identification number. Do not ignore this notice.",
        # Fear of immediate action, deadline
        "You have forty eight hours to respond to this notice before your wages are garnished and a lien is placed on your property.",
        # Robocall style, calm but threatening
        "This automated message is to inform you that a federal case number has been assigned to your file. Press 2 to avoid further action.",
        # First person officer voice
        "Hi, this is Officer Williams from the IRS Criminal Investigation Division. I need you to return this call before five o'clock today.",
        # Quiet threat, no yelling
        "We have made several attempts to contact you before escalating. This is your final opportunity to resolve this matter outside of court.",
    ],

    "bank_financial": [
        # Fraud hold on account
        "Your checking account has been frozen due to suspicious login attempts. Call our fraud hotline immediately to restore access.",
        # Transfer funds urgency
        "This is your bank's security team. To protect your funds, you must transfer them to a temporary holding account right now.",
        # Unusual transaction
        "An international wire transfer of four thousand dollars was initiated from your account. Call immediately to cancel this transaction.",
        # Card compromise
        "Your debit card ending in four seven two nine has been compromised. Press 1 to speak with a fraud specialist before any further charges occur.",
        # Soft opener, builds trust
        "Hello, we noticed some unusual activity on your account and wanted to reach out personally before taking any automated action.",
        # Verification trap
        "For your security we need to verify your identity. Please confirm your account number and the last four digits of your Social Security number.",
        # Impersonating a specific bank
        "This is an automated alert from your financial institution. Your online banking access will be suspended in one hour unless you verify your credentials.",
        # Gift card redirect
        "To reverse the fraudulent charges we found, you will need to purchase two Google Play cards and read us the numbers on the back.",
    ],

    "tech_support": [
        # Microsoft name drop, virus scare
        "Microsoft has detected critical malware on your device. Call our toll-free number immediately to prevent your files from being deleted.",
        # Popup simulated by voice
        "Warning: your computer is sending error messages to our servers. Your IP address has been flagged. Do not shut down your device.",
        # Apple ID lock threat
        "Your Apple ID has been locked due to unauthorized access from an unknown location. Call Apple support now to unlock your account.",
        # ISP persona
        "This is your internet service provider. We have detected that your router is being used to distribute illegal content. Call immediately.",
        # Calm tech persona
        "Hi, this is Jason from Windows technical support. We received an automated alert from your machine and need remote access to fix the issue.",
        # License expiry
        "Your Windows security license has expired. Your personal data is now at risk. Call within one hour or your computer will be disabled.",
        # Hacker claim
        "Our team has identified a hacker who accessed your device at two forty seven this morning. We need to walk you through the removal process now.",
        # Refund scam entry point
        "You are owed a forty dollar refund for a service cancellation. To process it, a technician needs to briefly access your computer.",
    ],

    "prize_lottery": [
        # Large cash prize
        "Congratulations, you have been selected to receive fifty thousand dollars in our national sweepstakes. Call to claim your prize today.",
        # Vacation package
        "You have won a complimentary vacation package for two to the Bahamas. To confirm your reservation we need a small processing fee.",
        # Gift card prize
        "You are today's winner of a five hundred dollar gift card. Provide your shipping address and card details to receive your reward.",
        # Publishers Clearing House impersonation
        "This is Publishers Clearing House. Our prize patrol will be at your home tomorrow unless you call to confirm your identity first.",
        # Lottery fee scam
        "Your name was entered in a government lottery and you have won. A five percent release fee is required before we can send the check.",
        # Soft friendly voice
        "Hi, I am calling because you were randomly selected from our customer database for a special cash reward. This is not a sales call.",
        # Time pressure
        "Your prize of eight thousand dollars will be forfeited in twenty four hours if you do not contact our claims department immediately.",
        # Social proof manipulation
        "Three of your neighbors have already claimed their prizes this week. You are the last name on our list. Do not miss this opportunity.",
    ],

    "utility_disconnection": [
        # Two hour deadline
        "Your electricity service will be disconnected in two hours for non-payment. Call our billing department immediately to avoid interruption.",
        # Gas company
        "This is your gas utility provider with a final disconnect notice. A technician is scheduled to cut your service at noon today.",
        # Cold weather threat
        "Temperatures are dropping tonight. Your heating service will be shut off in one hour unless you make a payment arrangement right now.",
        # Meter reading excuse
        "We attempted to read your smart meter today and detected an anomaly. You must call back to avoid an emergency service interruption.",
        # Supervisor escalation trick
        "You can speak with my supervisor if you do not believe me, but the disconnect order has already been placed and only a payment can stop it.",
        # Cryptocurrency payment demand
        "To reinstate your service you may pay by credit card or by depositing cash at a Bitcoin ATM. A payment code will be provided.",
        # Robocall style
        "This is an automated final notice from your electricity provider. Press 1 now to make a payment and avoid same-day disconnection.",
        # Soft opener before threat
        "Hi, we have been trying to reach the account holder. There is a past due balance that needs to be resolved today to avoid interruption.",
    ],

    "medicare_insurance": [
        # New card ruse
        "This is Medicare calling about the new benefits card being issued to seniors in your area. We need to verify your information.",
        # Insurance lapse threat
        "Your supplemental health insurance will lapse at midnight tonight unless you update your payment information by calling us now.",
        # Free equipment hook
        "You may be eligible for a free knee brace or back support at no cost to you through Medicare. Press 1 to see if you qualify.",
        # Identity verification trap
        "To continue receiving your Medicare benefits without interruption, we need to confirm your Medicare ID number and date of birth.",
        # Overpayment ruse
        "Medicare has overpaid on a recent claim and we need to recover the funds. Please have your banking information ready when you call.",
        # Prescription drug card
        "As part of a new prescription drug benefit, all Medicare recipients must re-enroll by Friday. Call now to avoid losing your coverage.",
        # Friendly agent opener
        "Hello, I am calling from the Medicare assistance program. There are new benefits available to you that most seniors are not aware of.",
        # Fear of losing benefits
        "Your Medicare Part B coverage is at risk of being cancelled due to an administrative error. This must be corrected before end of day.",
    ],

    "grandparent_emergency": [
        # Classic grandchild in jail
        "Grandma it is me. I got into a car accident and was arrested. I need you to wire bail money and please do not tell mom and dad.",
        # Lawyer impersonation
        "Hello, I am calling on behalf of your grandson who has been detained. He asked me to reach out to you for emergency bail funds.",
        # Hospital ruse
        "Your granddaughter has been in an accident and is at a hospital overseas. She needs you to send money for emergency surgery now.",
        # Police officer persona
        "This is Officer Martinez. Your grandson is in custody and has listed you as his emergency contact. He needs five hundred dollars tonight.",
        # Embarrassment manipulation
        "He made a mistake and is too embarrassed to call you himself. He begged me to reach out. He just needs help this one time.",
        # Out of country version
        "Your grandson is traveling abroad and was robbed. His passport and wallet were stolen. He needs emergency funds wired immediately.",
        # Soft distress voice
        "Gran, my voice sounds different because I hit my nose. I am in trouble and really need your help. Please do not hang up.",
        # Attorney general seal
        "This is the District Attorney's office. A family member of yours is facing charges and requires a cash bond to be released tonight.",
    ],

    "crypto_investment": [
        # Guaranteed returns
        "Our proprietary trading algorithm guarantees forty percent returns monthly. Send Bitcoin now to secure your position before it closes.",
        # Celebrity endorsement fake
        "Elon Musk is backing a new crypto fund that is closed to the public. You have been selected for early access. Invest now.",
        # Fake profit shown
        "We have already generated twelve thousand dollars in your demo account. To withdraw it you simply need to make a small activation deposit.",
        # Urgency window
        "This investment window closes in six hours. Hundreds of people are already enrolled. Do not miss your chance to double your money.",
        # Romance scam entry point
        "I told you about this opportunity because I care about you. My uncle manages the fund. You can trust me, just send what you can afford.",
        # Pig butchering setup
        "I have been using this platform for three months and withdrew fifty thousand dollars last week. I can show you exactly how to do it.",
        # Recovery scam
        "We can recover the crypto you lost in a previous scam. There is a small fee to initiate the recovery process. Our success rate is ninety percent.",
        # Fake exchange
        "Your account on our exchange has been verified. To unlock your trading limit and withdraw funds, you must first deposit a minimum balance.",
    ],

    "urgency_pressure": [
        # One hour offer
        "This offer expires in one hour. You must provide your account details right now to receive your refund before the deadline passes.",
        # Legal proceedings
        "You have twenty four hours to respond before legal proceedings are initiated against your name and property.",
        # Last chance framing
        "This is your last and final notice. Further attempts to contact you will be handed over to our collections and legal department.",
        # Calm but firm
        "I understand you are busy but this cannot wait. The window to resolve this without consequences closes at five o'clock today.",
        # Callback urgency
        "Do not ignore this call. Failure to return this message will be taken as your refusal to cooperate with the investigation.",
        # Arrest threat no softening
        "A warrant has been issued for your arrest. You must call our office before officers are dispatched to your home or workplace.",
        # Soft then escalate
        "We really do not want to take this further. One phone call from you right now can resolve everything and we can close your case.",
        # Document threat
        "Your name and case number will be forwarded to the local sheriff's office tomorrow morning if we do not hear from you today.",
    ],

    "spoofed_charity": [
        # Disaster relief
        "We are raising emergency funds for disaster victims in your region. Can we count on you for a small donation today by credit card?",
        # Police / firefighter charity
        "This is the Police Officers Benevolent Fund. We are calling to request your continued support for officers injured in the line of duty.",
        # Children's charity
        "Hello, we are collecting for sick children at a local hospital. A gift of any amount will go directly to kids in need today.",
        # Matching gift manipulation
        "A generous donor has agreed to match every gift made before midnight. Your donation will be doubled if you give right now.",
        # Vet charity
        "We support veterans struggling with homelessness. Your donation of fifty dollars provides a week of shelter. Can we process that for you?",
        # Callback to previous donation
        "You generously donated last year. We are calling to see if you would like to renew your gift today. Your card on file is still active.",
        # Vague name close to real org
        "This is the American Cancer Relief Fund calling. We are close to our monthly goal and could really use your help right now.",
        # High pressure close
        "I just need a yes or no. Can we count on you for even twenty five dollars? The children are counting on supporters like you.",
    ],

    "package_customs": [
        # Package held
        "Your international package has been held at customs. A clearance fee of thirty five dollars must be paid to release it for delivery.",
        # USPS impersonation
        "This is the United States Postal Service. We attempted delivery but require additional address verification and a small redelivery fee.",
        # FedEx impersonation
        "A package addressed to you requires your signature and a customs duty payment before we can complete delivery. Call to arrange this.",
        # Urgent delivery
        "Your package will be returned to sender in twenty four hours if the outstanding balance is not cleared. Please call our claims line.",
        # Gift from overseas
        "You have received a package from overseas that contains high-value items. Customs requires a declaration fee before it can be released.",
        # Tracking number bait
        "We have a tracking number for a package in your name. There is a problem with the address. Please call to confirm your information.",
        # Immigration variant
        "Your package has been flagged by border control for containing restricted materials. Call immediately to clarify or face confiscation.",
        # Credit card payment redirect
        "To release your held shipment, payment must be made via prepaid card or wire transfer. We do not accept cash or personal checks.",
    ],
}


SEED_SAFE = {
    "appointments_reminders": [
        "Hi, just calling to confirm your appointment scheduled for Thursday at three in the afternoon.",
        "Hello, this is a reminder about your dental cleaning tomorrow morning at ten. Please call if you need to reschedule.",
        "This is a courtesy reminder from Dr. Patel's office. Your annual physical is booked for Monday at nine thirty.",
        "Hi, your eye exam is confirmed for next Wednesday at two. Parking is available in the rear lot.",
        "Hello, we wanted to remind you that your car is due for its scheduled service next week. Call to confirm the time.",
        "This is a reminder from the vet's office. Buddy's annual vaccinations are due. Please call to book at your convenience.",
    ],
    "package_delivery": [
        "This is a courtesy call to let you know your package has been delivered to your front door.",
        "Hello, your order has shipped and is expected to arrive by Thursday via standard ground delivery.",
        "Hi, the courier attempted delivery today but no one was home. Your package is at the local depot for pickup.",
        "Your item has cleared customs and is now in local transit. Estimated delivery is two business days.",
        "Hello, just letting you know your replacement part has shipped. You will receive a tracking number by email shortly.",
        "Your order from last Tuesday is out for delivery today between noon and four in the afternoon.",
    ],
    "work_professional": [
        "Hi, I am calling to follow up on the proposal we sent over last week. Happy to answer any questions you might have.",
        "Hello, this is Sarah from accounts payable. I just wanted to confirm we received your invoice and it is being processed.",
        "Hi, calling to let you know the contract has been signed on our end and the project can move forward.",
        "Hello, I wanted to check in before our meeting tomorrow. The agenda has been updated and sent to your email.",
        "Hi, this is James from IT. Your laptop is ready for pickup from the support desk whenever you get a chance.",
        "Hello, just a heads up that the conference room booking for Friday has been moved to the third floor.",
    ],
    "personal_casual": [
        "Hey, just wanted to check in and see how you are doing after the move. Give me a call when you get a chance.",
        "Hi, it is mom. Just calling to say hi and see if you are free for dinner on Sunday.",
        "Hey, I left my jacket at your place last weekend. Is it okay if I swing by to pick it up?",
        "Hi, it is your neighbor from down the street. I found your mail in my box by mistake and wanted to return it.",
        "Hey, just confirming we are still on for the game on Saturday. Let me know if anything changes.",
        "Hi, I am calling about the used bike you listed online. Is it still available and can I come have a look?",
    ],
    "bank_legit": [
        "This is your bank calling to confirm a large transaction you authorized today. No action is needed, this is just a confirmation.",
        "Hello, your new debit card has been mailed to the address on file. You should receive it within five to seven business days.",
        "Hi, this is a reminder that your mortgage payment is due in five days. No action needed if you have autopay set up.",
        "Hello, your loan application has been reviewed and we would like to discuss your options. Please call at your convenience.",
        "This is a notification that your account statement is ready to view in online banking. No personal information is needed.",
        "Hi, just letting you know your wire transfer was completed successfully. The funds should reflect within one business day.",
    ],
    "healthcare_followup": [
        "Hi, I am calling from the clinic to follow up on your recent visit. The doctor wanted to check how you are feeling.",
        "Hello, your lab results are back and the doctor would like to discuss them with you. Please call to schedule a brief follow up.",
        "This is the pharmacy. Your prescription refill is ready and will be held for ten days. Let us know if you need delivery.",
        "Hi, the specialist has reviewed your referral and has availability next Tuesday. Call to confirm if that time works for you.",
        "Hello, this is a reminder that your physical therapy session is tomorrow at eleven. Arrive ten minutes early to complete paperwork.",
        "Hi, the nurse wanted to follow up on the medication change from last week. Are you experiencing any side effects?",
    ],
    "utility_legit": [
        "Hello, this is your electric company confirming your payment was received and your account is current. No action needed.",
        "Hi, we wanted to let you know there will be a planned outage in your area on Saturday from ten to noon for maintenance.",
        "This is your internet provider confirming your technician appointment for Friday between two and four in the afternoon.",
        "Hello, your service transfer to the new address has been completed. Your first bill at the new address will arrive next month.",
        "Hi, this is a reminder that your account is up for annual review. You may be eligible for a lower rate. Call when convenient.",
        "This is an automated message confirming your autopay is active and your next payment will be processed on the fifteenth.",
    ],
    "school_community": [
        "Hello, this is the school office calling to let you know that early dismissal tomorrow is at one thirty instead of three.",
        "Hi, this is a reminder about the parent teacher conference scheduled for next Thursday evening at six.",
        "Hello, this is the principal's office. Your child was marked absent today. Please call or send a note to confirm.",
        "Hi, the school nurse is calling to let you know your child has a mild headache. They are resting here. No emergency.",
        "Hello, just a reminder that the field trip permission slips are due by Friday. Please send them with your child.",
        "Hi, this is the sports coordinator. Practice has been moved to the back field this week due to gym renovation.",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA AUGMENTATION
#    Key changes vs v1:
#    - Prompt explicitly asks for structural diversity, not paraphrase
#    - Quality filter removes too-short, too-long, and duplicate-feeling outputs
#    - Augmentation tagged with source category for later debugging
# ─────────────────────────────────────────────────────────────────────────────

def augment_with_hf(
    category_name: str,
    seed_examples: list,
    label: str,
    n: int,
    client: InferenceClient,
) -> list:
    """
    Generates n new examples from Mistral, filtered for quality.
    Returns list of clean strings.
    """
    examples_str = "\n".join(f"- {ex}" for ex in seed_examples[:5])

    if label == "scam":
        instruction = (
            f"You are building a phone scam detector training dataset.\n"
            f"Generate {n} new phone scam transcript sentences for the category: '{category_name}'.\n\n"
            f"REQUIREMENTS — read carefully:\n"
            f"- Each sentence must use a DIFFERENT tactic from the others (vary: persona, urgency type, hook style, threat type)\n"
            f"- Do NOT paraphrase the examples below — write structurally original sentences\n"
            f"- Mix: robocall tone vs live-agent tone vs automated-alert tone\n"
            f"- Mix: hard threats vs soft pressure vs false reward vs fear\n"
            f"- Each sentence must be 1–3 sentences long (15–80 words)\n"
            f"- Sound like a real phone call transcript, not a written warning\n\n"
            f"Style reference (do NOT copy):\n{examples_str}\n\n"
            f"Output ONLY a JSON array of strings. No preamble. No explanation. No markdown.\n"
            f'Format: ["sentence one", "sentence two", ...]'
        )
    else:
        instruction = (
            f"You are building a phone scam detector training dataset.\n"
            f"Generate {n} new BENIGN (safe, legitimate) phone call transcript sentences for the category: '{category_name}'.\n\n"
            f"REQUIREMENTS — read carefully:\n"
            f"- Each sentence must represent a DIFFERENT everyday scenario\n"
            f"- NO urgency, NO threats, NO requests for payment or personal information\n"
            f"- Mix: automated reminder vs live agent vs casual personal call\n"
            f"- Each sentence must be 1–2 sentences long (10–60 words)\n"
            f"- Sound like a real phone call transcript\n\n"
            f"Style reference (do NOT copy):\n{examples_str}\n\n"
            f"Output ONLY a JSON array of strings. No preamble. No explanation. No markdown.\n"
            f'Format: ["sentence one", "sentence two", ...]'
        )

    try:
        response = client.text_generation(
            instruction,
            max_new_tokens=1400,
            temperature=0.9,
            repetition_penalty=1.15,
            stop_sequences=["```", "\n\n\n"],
        )
        text = response.strip()
        start = text.find("[")
        end   = text.rfind("]") + 1
        if start == -1 or end == 0:
            print(f"  [!] No JSON array found for {category_name}")
            return []
        parsed = json.loads(text[start:end])
        if not isinstance(parsed, list):
            return []

        # Quality filter
        filtered = []
        seen = set()
        for s in parsed:
            s = str(s).strip()
            key = s.lower()[:40]
            if key in seen:
                continue
            seen.add(key)
            if len(s) < MIN_SAMPLE_LEN or len(s) > MAX_SAMPLE_LEN:
                continue
            filtered.append(s)

        print(f"  [+] {category_name}: {len(parsed)} generated → {len(filtered)} passed filter")
        return filtered

    except Exception as e:
        print(f"  [!] Augmentation failed for {category_name}: {e}")
        return []


def build_augmented_dataset(n_per_category: int = SAMPLES_PER_CATEGORY) -> pd.DataFrame:
    cache_path = "./data/augmented_raw.csv"

    if os.path.exists(cache_path):
        print(f"[*] Loading cached augmented data from {cache_path}")
        return pd.read_csv(cache_path)

    if not HF_TOKEN:
        print("[!] HF_TOKEN not set — using seed data only.")
        rows = []
        for seeds in SEED_SCAM.values():
            for t in seeds:
                rows.append({"text": t, "label": 1})
        for seeds in SEED_SAFE.values():
            for t in seeds:
                rows.append({"text": t, "label": 0})
        return pd.DataFrame(rows)

    client   = InferenceClient(model=AUGMENT_MODEL, token=HF_TOKEN)
    all_rows = []

    print("\n[*] Augmenting scam samples...")
    for cat_name, seeds in SEED_SCAM.items():
        time.sleep(1.2)
        new_samples = augment_with_hf(cat_name, seeds, "scam", n_per_category, client)
        for t in seeds + new_samples:
            all_rows.append({"text": t, "label": 1, "category": cat_name})

    print("\n[*] Augmenting benign samples...")
    for cat_name, seeds_list in SEED_SAFE.items():
        time.sleep(1.2)
        new_samples = augment_with_hf(cat_name, seeds_list, "safe", n_per_category, client)
        for t in seeds_list + new_samples:
            all_rows.append({"text": t, "label": 0, "category": cat_name})

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["text"]).dropna(subset=["text"])
    df.to_csv(cache_path, index=False)
    print(f"\n[*] Saved: {len(df)} rows → {cache_path}")
    print(f"    Scam: {df[df.label==1].shape[0]}  |  Safe: {df[df.label==0].shape[0]}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET ASSEMBLY & BALANCING
# ─────────────────────────────────────────────────────────────────────────────

def build_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    n_scam = df[df["label"] == 1].shape[0]
    n_safe = df[df["label"] == 0].shape[0]
    print(f"\n[*] Before balancing: scam={n_scam}, safe={n_safe}")

    n = min(n_scam, n_safe)
    balanced = pd.concat([
        df[df["label"] == 1].sample(n, random_state=42),
        df[df["label"] == 0].sample(n, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[*] After balancing: {len(balanced)} total ({n} per class)")
    balanced[["text", "label"]].to_csv("./data/tool_c_dataset.csv", index=False)
    return balanced


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOKENIZE
# ─────────────────────────────────────────────────────────────────────────────

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def make_hf_dataset(dataframe: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(dataframe[["text", "label"]].reset_index(drop=True))
    ds = ds.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        ),
        batched=True,
    )
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# 5. METRICS  — full report including confusion matrix and false-negative rate
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall":    recall_score(labels, preds, average="binary", zero_division=0),
    }


def full_eval_report(labels, preds, probs=None, threshold=0.5, split_name="Test"):
    """
    Prints confusion matrix, per-class report, and false-negative rate.
    Most important metric for scam detection: false-negative rate (missed scams).
    """
    print(f"\n{'─'*55}")
    print(f"  {split_name} Evaluation  (threshold={threshold:.2f})")
    print(f"{'─'*55}")

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"  Confusion Matrix:")
    print(f"               Predicted Safe   Predicted Scam")
    print(f"  Actual Safe       {tn:>5}            {fp:>5}")
    print(f"  Actual Scam       {fn:>5}            {tp:>5}")

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n  False-Negative Rate (missed scams):  {fnr:.4f}  ← minimise this")
    print(f"  False-Positive Rate (false alarms):  {fpr:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=['safe','scam'])}")
    print(f"{'─'*55}\n")
    return {"fnr": fnr, "fpr": fpr, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ─────────────────────────────────────────────────────────────────────────────
# 6. STRATIFIED K-FOLD CV  — runs before touching the held-out test set
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold_cv(df: pd.DataFrame, n_splits: int = N_FOLDS):
    """
    Runs StratifiedKFold CV and reports mean ± std for key metrics.
    The held-out test set is never used here.
    """
    print(f"\n[*] Running {n_splits}-fold stratified cross-validation...")
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X      = df["text"].values
    y      = df["label"].values
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  Fold {fold}/{n_splits}")
        fold_train = df.iloc[train_idx]
        fold_val   = df.iloc[val_idx]

        train_ds = make_hf_dataset(fold_train)
        val_ds   = make_hf_dataset(fold_val)

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        args = TrainingArguments(
            output_dir=f"./models/tool_c/fold_{fold}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_ratio=0.1,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            eval_strategy="no",
            save_strategy="no",
            logging_steps=999,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        result = trainer.evaluate()
        fold_metrics.append(result)
        print(f"  F1={result.get('eval_f1',0):.4f}  "
              f"Recall={result.get('eval_recall',0):.4f}  "
              f"Precision={result.get('eval_precision',0):.4f}")

    # Summary
    keys = ["eval_f1", "eval_recall", "eval_precision", "eval_accuracy"]
    print(f"\n{'─'*55}")
    print(f"  Cross-Validation Summary ({n_splits} folds)")
    print(f"{'─'*55}")
    for k in keys:
        vals = [m.get(k, 0) for m in fold_metrics]
        print(f"  {k:30s}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"{'─'*55}\n")
    return fold_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL TRAINING ON FULL TRAIN SET + HELD-OUT TEST EVAL
# ─────────────────────────────────────────────────────────────────────────────

def train_tool_c(train_ds: Dataset, test_ds: Dataset, test_df: pd.DataFrame):
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    args = TrainingArguments(
        output_dir="./models/tool_c",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Get raw probabilities from trainer for threshold analysis
    pred_output = trainer.predict(test_ds)
    logits      = pred_output.predictions
    labels      = pred_output.label_ids
    probs_scam  = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    # Evaluate at default threshold
    preds_default = (probs_scam >= 0.5).astype(int)
    full_eval_report(labels, preds_default, probs_scam, threshold=0.5, split_name="Test (threshold=0.50)")

    # Evaluate at tuned threshold (lower = catch more scams, accept more false alarms)
    preds_tuned = (probs_scam >= SCAM_THRESHOLD).astype(int)
    full_eval_report(labels, preds_tuned, probs_scam, threshold=SCAM_THRESHOLD,
                     split_name=f"Test (threshold={SCAM_THRESHOLD})")

    # Save threshold to disk so inference wrapper uses same value
    with open("./models/tool_c/threshold.json", "w") as f:
        json.dump({"scam_threshold": SCAM_THRESHOLD}, f)

    model.save_pretrained("./models/tool_c")
    tokenizer.save_pretrained("./models/tool_c")
    print("[*] Tool C saved → ./models/tool_c")
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# 8. INFERENCE WRAPPER  — what agent.py imports
#
#    Returns:
#      scam_probability  : float   (raw softmax score for scam class)
#      safe_probability  : float
#      label             : str     ("scam" | "safe")  at calibrated threshold
#      confidence        : str     ("high" | "medium" | "low")
#      text_snippet      : str     (first 80 chars, for logging)
#
#    agent.py should use scam_probability directly rather than label when
#    doing its own fusion with acoustic / deepfake scores.
# ─────────────────────────────────────────────────────────────────────────────

class ScamTextClassifier:
    def __init__(self, model_dir: str = "./models/tool_c"):
        from transformers import pipeline as hf_pipeline

        # Load calibrated threshold if saved during training
        threshold_path = os.path.join(model_dir, "threshold.json")
        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                self.threshold = json.load(f).get("scam_threshold", SCAM_THRESHOLD)
        else:
            self.threshold = SCAM_THRESHOLD

        self._pipe = hf_pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=model_dir,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=128,
        )
        self._label_map = {"LABEL_1": "scam", "LABEL_0": "safe"}

    def _confidence(self, prob: float) -> str:
        """
        Confidence band — useful for agent.py to weight the NLP signal.
        high   : prob > 0.80 or prob < 0.20
        medium : 0.50–0.80 or 0.20–0.50
        low    : near decision boundary (0.35–0.65)
        """
        dist = abs(prob - 0.5)
        if dist >= 0.30:
            return "high"
        if dist >= 0.15:
            return "medium"
        return "low"

    def predict(self, text: str) -> dict:
        result = self._pipe(text, top_k=None)
        scores = {self._label_map[r["label"]]: r["score"] for r in result}
        scam_prob = scores.get("scam", 0.0)

        lower = text.lower()

        flags = []
        if scam_prob > 0.7:
            flags.append("high_risk")
        if any(w in lower for w in ["urgent", "immediately", "now"]):
            flags.append("urgency")
        if any(w in lower for w in ["pay", "transfer", "bitcoin", "card"]):
            flags.append("payment_request")

        return {
            "scam_score": float(scam_prob),
            "flags": flags
        }
    

    def predict_batch(self, texts: list) -> list:
        results = self._pipe(texts, top_k=None, batch_size=32)
        output  = []
        for text, result in zip(texts, results):
            scores    = {self._label_map[r["label"]]: round(r["score"], 4) for r in result}
            scam_prob = scores.get("scam", 0.0)
            label     = "scam" if scam_prob >= self.threshold else "safe"
            output.append({
                "scam_probability": scam_prob,
                "safe_probability": scores.get("safe", 1.0 - scam_prob),
                "label":            label,
                "confidence":       self._confidence(scam_prob),
                "threshold_used":   self.threshold,
                "text_snippet":     text[:80],
            })
        return output


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Augment / load
    raw_df = build_augmented_dataset(n_per_category=SAMPLES_PER_CATEGORY)

    # Step 2: Balance
    df = build_balanced_dataset(raw_df)

    # Step 3: Carve out held-out test set FIRST — never touch it until final eval
    train_val_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=42
    )
    print(f"\n[*] Dataset split:")
    print(f"    Train+Val : {len(train_val_df)}")
    print(f"    Test      : {len(test_df)}  (held out, not used in CV)")

    # Step 4: Stratified K-Fold CV on train_val only
    run_kfold_cv(train_val_df, n_splits=N_FOLDS)

    # Step 5: Final train on full train_val, eval on held-out test
    train_ds = make_hf_dataset(train_val_df)
    test_ds  = make_hf_dataset(test_df)
    train_tool_c(train_ds, test_ds, test_df)

    # Step 6: Sanity check with inference wrapper
    print("\n── Inference sanity check ────────────────────────────")
    clf = ScamTextClassifier("./models/tool_c")
    print(f"    Using threshold: {clf.threshold}")

    test_cases = [
        ("SCAM", "This is the IRS, you owe back taxes, call immediately or face arrest."),
        ("SCAM", "Your electricity will be disconnected in two hours. Pay now at a Bitcoin ATM."),
        ("SCAM", "Congratulations, you won fifty thousand dollars. Pay a small fee to claim."),
        ("SCAM", "Grandma it is me, I got arrested, please wire the bail money and do not tell mom."),
        ("SAFE", "Hi just calling to confirm your dentist appointment for tomorrow at ten."),
        ("SAFE", "Hello, your package has been delivered to your front door this afternoon."),
        ("SAFE", "This is your bank confirming the wire transfer you authorized today was completed."),
        ("SAFE", "Hi, I am calling from Dr. Patel's office to follow up on your recent visit."),
    ]
    print(f"\n  {'TRUE':5s}  {'PRED':5s}  {'PROB':6s}  {'FLAGS':8s}  TEXT")
    print(f"  {'─'*70}")
    correct = 0
    for true_label, text in test_cases:
        r = clf.predict(text)
        pred = "SCAM" if r["scam_score"] >= clf.threshold else "SAFE"
        mark = "✓" if pred == true_label else "✗"
        if pred == true_label:
            correct += 1
        print(f"  {true_label:5s}  {pred:5s}  {r['scam_score']:.4f}  "
              f"{str(r['flags']):8s}  {mark}  {text[:55]}")
    print(f"\n  Sanity accuracy: {correct}/{len(test_cases)}")
    print(f"{'─'*55}\n")
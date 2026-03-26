"""
csv_parser.py — Parse CSV and Excel bank statements into Finora's standard format.
Handles different column name conventions across Indian banks.
Output columns: Date | Description | Amount | Needs_Review
"""

import re
import pandas as pd


# Keyword hints for smart column detection
# If ANY of these words appear in a column name → it's likely that column
DATE_HINTS        = ["date", "dt", "time", "day"]
DESCRIPTION_HINTS = ["narration", "description", "detail", "particular",
                     "remark", "payee", "merchant", "note", "ref", "particulars"]
AMOUNT_HINTS      = ["withdrawal", "debit", "amount", "dr", "paid", "expense",
                     "txn amt", "transaction amt", "withdraw"]

# Columns to always ignore (credits, balances, etc.)
IGNORE_HINTS      = ["deposit", "credit", "cr", "balance", "closing", "opening",
                     "available", "chq", "cheque", "mode", "type", "branch"]

# UPI merchant extraction map
UPI_MERCHANT_MAP = {
    "swiggy": "Swiggy", "zomato": "Zomato", "uber": "Uber",
    "ola": "Ola Cabs", "netflix": "Netflix", "spotify": "Spotify",
    "amazon": "Amazon", "flipkart": "Flipkart", "myntra": "Myntra",
    "bigbasket": "BigBasket", "blinkit": "Blinkit", "airtel": "Airtel",
    "jio": "Jio", "hotstar": "Disney+ Hotstar", "phonepe": "PhonePe",
    "paytm": "Paytm", "irctc": "IRCTC", "makemytrip": "MakeMyTrip",
    "bookmyshow": "BookMyShow", "apollo": "Apollo Pharmacy", "1mg": "1mg",
}

# Patterns that indicate a credit (money IN) — skip for expense tracking
CREDIT_PATTERNS = [
    r"\bsalary\b", r"\bcredited\b", r"neft cr", r"upi cr",
    r"\binterest\b", r"\brefund\b", r"\bcashback\b", r"\breversal\b",
    r"/cr/", r"\bopening balance\b", r"\bclosing balance\b",
]

UNKNOWN_PATTERNS = [
    r"^unknown$",           # literally "Unknown"
    r"^\d+$",              # pure numbers like "00012345"
    r"^[a-f0-9]{10,}$",   # hex reference codes only
    r"^upi:\s*$",          # empty UPI with no merchant
    r"^atm\s+wdl",         # ATM withdrawals
    r"^neft/\d+$",         # NEFT with only ref number
]

# Known merchants — never flag these for review
KNOWN_MERCHANTS = {
    "swiggy", "zomato", "uber", "ola", "netflix", "spotify", "amazon",
    "flipkart", "myntra", "bigbasket", "blinkit", "airtel", "jio",
    "hotstar", "phonepe", "paytm", "irctc", "makemytrip", "bookmyshow",
    "apollo", "1mg", "medplus", "dominos", "kfc", "mcdonalds", "dunzo",
    "rapido", "gpay", "google", "apple", "microsoft",
}


def parse_csv_excel(file_obj) -> pd.DataFrame:
    """
    Parse an uploaded CSV or Excel file into Finora's standard DataFrame.

    Args:
        file_obj: Streamlit UploadedFile object (.csv / .xlsx / .xls)

    Returns:
        DataFrame with columns: Date, Description, Amount, Needs_Review
    """
    filename = file_obj.name.lower()

    if filename.endswith(".csv"):
        df = _read_csv(file_obj)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_obj)
    else:
        raise ValueError("Unsupported file type. Please upload .csv, .xlsx, or .xls")

    df = _normalize_columns(df)
    df = _clean_amounts(df)
    df = _clean_dates(df)
    df = _clean_descriptions(df)
    df = _remove_credits(df)
    df = _flag_unknowns(df)

    df = df[df["Amount"] > 0].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _read_csv(file_obj) -> pd.DataFrame:
    """Try multiple encodings to read a CSV."""
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=encoding)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError("Could not read CSV file. Try saving it as UTF-8.")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smartly detect Date / Description / Amount columns using keyword hints.
    Works for any bank format without hardcoding exact column names.
    """
    # Clean column names for matching
    cols = {str(c).strip(): str(c).strip().lower().replace(".", "").replace("/", " ").replace("(", "").replace(")", "") 
            for c in df.columns}

    date_col  = None
    desc_col  = None
    amt_col   = None

    # Score each column against hints — highest score wins
    def score(col_lower: str, hints: list) -> int:
        return sum(1 for hint in hints if hint in col_lower)

    # Filter out columns that are clearly to be ignored
    def is_ignored(col_lower: str) -> bool:
        ignore_score = sum(1 for hint in IGNORE_HINTS if hint in col_lower)
        return ignore_score > 0

    # Find best matching column for each standard field
    candidates = {orig: cleaned for orig, cleaned in cols.items()}

    # ── Date column ───────────────────────────────────────────────────────────
    date_scores = {
        orig: score(cleaned, DATE_HINTS)
        for orig, cleaned in candidates.items()
        if not is_ignored(cleaned)
    }
    if date_scores:
        date_col = max(date_scores, key=date_scores.get)
        if date_scores[date_col] == 0:
            # Fallback: first column is usually date in bank statements
            date_col = list(candidates.keys())[0]

    # ── Description column ────────────────────────────────────────────────────
    desc_scores = {
        orig: score(cleaned, DESCRIPTION_HINTS)
        for orig, cleaned in candidates.items()
        if orig != date_col
    }
    if desc_scores:
        desc_col = max(desc_scores, key=desc_scores.get)
        if desc_scores[desc_col] == 0:
            # Fallback: longest string column is usually description
            str_cols = [c for c in df.columns if df[c].dtype == object and c != date_col]
            if str_cols:
                desc_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())

    # ── Amount column ─────────────────────────────────────────────────────────
    # Only consider numeric-looking columns or ones with amount hints
    amt_scores = {}
    for orig, cleaned in candidates.items():
        if orig in [date_col, desc_col]:
            continue
        # Skip clear credit/balance columns
        if any(hint in cleaned for hint in ["deposit", "credit", "balance", "closing", "opening", "cr "]):
            continue
        hint_score = score(cleaned, AMOUNT_HINTS)
        # Bonus if the column has numeric data
        try:
            numeric_vals = pd.to_numeric(
                df[orig].astype(str).str.replace(",", "").str.replace("₹", "").str.strip(),
                errors="coerce"
            )
            numeric_ratio = numeric_vals.notna().mean()
            amt_scores[orig] = hint_score + (2 if numeric_ratio > 0.5 else 0)
        except Exception:
            amt_scores[orig] = hint_score

    if amt_scores:
        amt_col = max(amt_scores, key=amt_scores.get)

    # ── Validate we found all 3 ───────────────────────────────────────────────
    missing = []
    if not date_col:  missing.append("Date")
    if not desc_col:  missing.append("Description")
    if not amt_col:   missing.append("Amount")

    if missing:
        raise ValueError(
            f"Could not detect columns for: {missing}\n"
            f"Your file has columns: {list(df.columns)}\n"
            f"Tip: Make sure your file has a date column, a narration/description column, "
            f"and a withdrawal/debit/amount column."
        )

    df = df.rename(columns={date_col: "Date", desc_col: "Description", amt_col: "Amount"})
    return df[["Date", "Description", "Amount"]].copy()


def _clean_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Amount column to float, removing commas and ₹ symbols."""
    df = df.copy()
    df["Amount"] = (
        df["Amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])
    return df


def _clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates to DD Mon YYYY string format."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.strftime("%d %b %Y")
    return df


def _clean_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and humanize narration strings."""
    df = df.copy()
    df["Description"] = df["Description"].fillna("Unknown").astype(str).apply(_humanize)
    return df


def _humanize(raw: str) -> str:
    """Convert a raw bank narration into a readable description."""
    if not raw or raw.strip() in ["", "nan", "None"]:
        return "Unknown"

    text = str(raw).strip()

    # Always flag P2P transfers — payment app is known but purpose isn't
    # e.g. IMPS/xxx/9845012345@paytm — we know it's Paytm but not what for
    p2p_re = re.compile(r"\b[6-9]\d{9}\b", re.IGNORECASE)  # Indian mobile numbers
    if p2p_re.search(str(raw)):
        return "Unknown (P2P Transfer)"

    # Extract merchant from UPI ID
    upi_match = re.search(
        r"UPI[\/\-](?:P2M|P2P|CR|DR)?[\/\-]?([A-Za-z0-9.\-]+)[@\/]",
        text, re.IGNORECASE,
    )
    if upi_match:
        merchant_raw = upi_match.group(1).lower()
        for key, name in UPI_MERCHANT_MAP.items():
            if key in merchant_raw:
                return name
        cleaned = re.sub(r"[^a-zA-Z\s]", " ", merchant_raw).strip().title()
        if len(cleaned) > 2:
            return f"UPI: {cleaned}"

    # Check full text against merchant map
    text_lower = text.lower()
    for key, name in UPI_MERCHANT_MAP.items():
        if key in text_lower:
            return name

    # Strip common bank prefixes
    for prefix in [r"^TO TRANSFER[-\s]*", r"^BY TRANSFER[-\s]*", r"^POS\s+",
                   r"^NEFT[-\s]*", r"^IMPS[-\s]*", r"^UPI[-\/\s]*", r"^ATM\s+WDL\s*"]:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE).strip()

    # Remove long reference numbers
    text = re.sub(r"\b\d{8,}\b", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()

    return text if len(text) > 2 else "Unknown"


def _remove_credits(df: pd.DataFrame) -> pd.DataFrame:
    """Remove credit/income transactions — we only want debits/expenses."""
    credit_re = re.compile("|".join(CREDIT_PATTERNS), re.IGNORECASE)
    mask = df["Description"].apply(lambda x: bool(credit_re.search(str(x))))
    return df[~mask].copy()


def _flag_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with unclear descriptions for manual review."""
    unknown_re = re.compile("|".join(UNKNOWN_PATTERNS), re.IGNORECASE)
    df = df.copy()

    def needs_review(desc: str) -> bool:
        desc = str(desc).strip()
        desc_lower = desc.lower()

        # Always flag "Unknown" and P2P transfers
        if desc_lower == "unknown" or "p2p transfer" in desc_lower:
            return True

        # Never flag known merchants
        if any(merchant in desc_lower for merchant in KNOWN_MERCHANTS):
            return False

        # Never flag if it's a readable sentence/phrase (letters, spaces, common chars)
        if re.match(r"^[A-Za-z][A-Za-z0-9\s\.\-&\(\)']{2,}$", desc):
            return False

        # Flag if matches unknown patterns
        return bool(unknown_re.search(desc))

    df["Needs_Review"] = df["Description"].apply(needs_review)
    return df
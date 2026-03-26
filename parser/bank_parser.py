"""
bank_parser.py — Smart Bank Statement Parser
Supports: HDFC, SBI (and generic fallback for other banks)

Key features:
  - Bank auto-detection from PDF content
  - HDFC-specific table extraction
  - SBI-specific table extraction
  - UPI narration cleaning (extracts merchant from UPI IDs)
  - Flags unknown/unrecognizable narrations for manual labeling
  - Generic fallback parser for other bank formats
  - OCR fallback for scanned/image-based PDFs (requires pytesseract)
"""

import re
import pdfplumber
import pandas as pd


# ─────────────────────────────────────────────
#  OCR availability check
# ─────────────────────────────────────────────

def _ocr_available() -> bool:
    """Check if pytesseract, pdf2image AND poppler are all available."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError
        # Quick test — try converting nothing, just check poppler exists
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _extract_text_via_ocr(pdf_path: str) -> list | None:
    """
    Fallback: convert PDF pages to images and run OCR.
    Returns list of text lines, or None if any dependency is missing/broken.

    Requires:
      pip install pytesseract pdf2image
      brew install tesseract poppler   (Mac)
      sudo apt install tesseract-ocr poppler-utils   (Linux)
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError

        lines = []
        try:
            images = convert_from_path(pdf_path, dpi=300)
        except PDFInfoNotInstalledError:
            print("OCR not available: Poppler is not installed. "
                  "Install with: brew install poppler")
            return None
        except Exception as e:
            print(f"OCR pdf2image error: {e}")
            return None

        for image in images:
            try:
                text = pytesseract.image_to_string(image, lang="eng")
                if text:
                    lines.extend(text.split("\n"))
            except Exception as e:
                print(f"OCR tesseract error on page: {e}")
                continue

        return lines if lines else None

    except ImportError:
        print("OCR not available: pytesseract or pdf2image not installed.")
        return None
    except Exception as e:
        print(f"OCR not available, skipping fallback: {e}")
        return None


def _is_scanned_pdf(raw_text: list) -> bool:
    """
    Detect if a PDF is scanned (image-based) by checking
    if pdfplumber extracted any meaningful text.
    """
    meaningful = [l for l in raw_text if len(l.strip()) > 5]
    return len(meaningful) == 0


def parse_bank_statement(pdf_path: str) -> pd.DataFrame:
    """
    Robust hybrid parser — never crashes.

    Strategy:
      1. pdfplumber table extraction (best for structured bank PDFs)
      2. pdfplumber text parsing (fallback for unstructured PDFs)
      3. OCR (only if PDF is scanned AND all OCR deps are available)
      4. Return empty DataFrame with helpful flag if all methods fail

    Returns DataFrame with columns:
      Date | Description | Amount | Needs_Review | Extraction_Method
    """
    empty_df = pd.DataFrame(
        columns=["Date", "Description", "Amount", "Needs_Review"]
    )
    used_ocr = False

    # ── Step 1 & 2: Try pdfplumber ────────────────────────────────────────────
    try:
        raw_text, tables = _extract_pdf_content(pdf_path)
    except Exception as e:
        print(f"pdfplumber failed: {e}")
        raw_text, tables = [], []

    # ── Step 3: OCR fallback if PDF is scanned ────────────────────────────────
    if _is_scanned_pdf(raw_text):
        print("PDF appears to be scanned — attempting OCR fallback...")
        ocr_lines = _extract_text_via_ocr(pdf_path)

        if ocr_lines:
            raw_text = ocr_lines
            tables   = []
            used_ocr = True
        else:
            # OCR unavailable or failed — return empty with flag
            print("All extraction methods failed. PDF may need Poppler+Tesseract.")
            df = empty_df.copy()
            df["_ocr_needed"] = True
            return df

    # ── Detect bank and parse ─────────────────────────────────────────────────
    try:
        bank = _detect_bank(raw_text)
        print(f"[PDF] Detected bank: {bank}")

        if bank == "HDFC":
            df = _parse_hdfc(raw_text, tables)
        elif bank == "SBI":
            df = _parse_sbi(raw_text, tables)
        else:
            # Generic: try smart table parser first, then text fallback
            generic_txns = _parse_tables_smart(tables)
            if generic_txns:
                df = _to_dataframe(generic_txns)
            else:
                df = _parse_generic(raw_text)
    except Exception as e:
        print(f"Parsing error: {e}")
        return empty_df

    if df.empty:
        return empty_df

    # ── Clean, flag, finalize ─────────────────────────────────────────────────
    try:
        df = _clean_descriptions(df)
        df = _flag_unknown_narrations(df)
    except Exception as e:
        print(f"Cleaning error: {e}")

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df.dropna(subset=["Amount"], inplace=True)
    df = df[df["Amount"] > 0]
    df.reset_index(drop=True, inplace=True)
    df["Extraction_Method"] = "OCR" if used_ocr else "pdfplumber"

    return df


# ─────────────────────────────────────────────
#  PDF content extraction
# ─────────────────────────────────────────────

def _extract_pdf_content(pdf_path: str):
    """Extract both raw text and ALL tables from every page."""
    raw_text = []
    tables   = []
    with pdfplumber.open(pdf_path) as pdf:
        print(f"[PDF] Total pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                raw_text.extend(text.split("\n"))
            # Try both extract_table and extract_tables
            tbls = page.extract_tables()
            if tbls:
                tables.extend(tbls)
                print(f"[PDF] Page {i+1}: {len(tbls)} table(s) found")
            elif page.extract_table():
                tables.append(page.extract_table())
    print(f"[PDF] Total tables found: {len(tables)}")
    return raw_text, tables


# ─────────────────────────────────────────────
#  Bank detection
# ─────────────────────────────────────────────

def _detect_bank(lines: list) -> str:
    """Identify bank from header text."""
    combined = " ".join(lines[:30]).lower()
    if "hdfc" in combined:
        return "HDFC"
    if "state bank of india" in combined or "sbi" in combined:
        return "SBI"
    return "GENERIC"


# ─────────────────────────────────────────────
#  Smart Table Parser (replaces HDFC/SBI fixed-column parsers)
# ─────────────────────────────────────────────

def _parse_tables_smart(tables: list) -> list:
    """
    Smart table parser — detects column positions dynamically.
    Handles any layout with Date | Description | Debit/Credit/Amount columns.
    Converts Debit/Credit → single Amount (debit = expense).
    """
    DATE_HINTS   = ["date", "dt", "txn date", "trans date", "value date"]
    DESC_HINTS   = ["narration", "description", "particulars", "details", "remarks", "payee"]
    DEBIT_HINTS  = ["withdrawal", "debit", "dr", "debit amt", "withdrawal amt"]
    CREDIT_HINTS = ["deposit", "credit", "cr", "credit amt"]
    AMOUNT_HINTS = ["amount", "txn amount", "transaction amount"]

    transactions = []

    for table in tables:
        if not table or len(table) < 2:
            continue

        # ── Detect header row ─────────────────────────────────────────────────
        header_idx = None
        col_map    = {}

        for row_idx, row in enumerate(table[:5]):  # header is usually in first 5 rows
            if not row:
                continue
            row_lower = [str(c).strip().lower() if c else "" for c in row]

            date_col  = next((i for i, c in enumerate(row_lower) if any(h in c for h in DATE_HINTS)), None)
            desc_col  = next((i for i, c in enumerate(row_lower) if any(h in c for h in DESC_HINTS)), None)
            debit_col = next((i for i, c in enumerate(row_lower) if any(h in c for h in DEBIT_HINTS)), None)
            credit_col= next((i for i, c in enumerate(row_lower) if any(h in c for h in CREDIT_HINTS)), None)
            amt_col   = next((i for i, c in enumerate(row_lower) if any(h in c for h in AMOUNT_HINTS)), None)

            if date_col is not None and desc_col is not None:
                header_idx = row_idx
                col_map = {
                    "date":   date_col,
                    "desc":   desc_col,
                    "debit":  debit_col,
                    "credit": credit_col,
                    "amount": amt_col,
                }
                break

        if header_idx is None:
            # No header found — try parsing with first col=date, second=desc, last=amount
            col_map = {"date": 0, "desc": 1, "debit": None, "credit": None, "amount": -1}
            header_idx = 0

        print(f"[TABLE] Header at row {header_idx}, col_map={col_map}")
        rows_processed = 0
        rows_extracted = 0

        # ── Parse data rows ───────────────────────────────────────────────────
        for row in table[header_idx + 1:]:
            if not row:
                continue
            rows_processed += 1

            # Get date
            date_val = str(row[col_map["date"]]).strip() if row[col_map["date"]] else ""
            if not date_val or not _is_date(date_val):
                continue

            # Get description
            desc_val = str(row[col_map["desc"]]).strip() if row[col_map["desc"]] else "Unknown"

            # Get amount — prefer debit col, fallback to amount col
            amount = None

            if col_map.get("debit") is not None:
                debit  = _parse_amount_cell(row[col_map["debit"]])
                credit = _parse_amount_cell(row[col_map["credit"]]) if col_map.get("credit") is not None else 0
                # Only include debits (expenses), skip credits (salary/refunds)
                if debit and debit > 0:
                    amount = debit
                elif credit and credit > 0:
                    continue  # skip credits

            if amount is None and col_map.get("amount") is not None:
                idx = col_map["amount"]
                if idx == -1:
                    idx = len(row) - 1
                amount = _parse_amount_cell(row[idx])

            # Last resort — scan all cells for a numeric value
            if amount is None or amount <= 0:
                for cell in row:
                    val = _parse_amount_cell(cell)
                    if val and val > 0:
                        amount = val
                        break

            if not amount or amount <= 0:
                continue

            transactions.append([date_val, desc_val, amount])
            rows_extracted += 1

        print(f"[TABLE] Rows processed: {rows_processed}, extracted: {rows_extracted}")

    return transactions


def _parse_amount_cell(cell) -> float:
    """Safely parse a table cell into a float amount."""
    if cell is None:
        return 0.0
    try:
        cleaned = str(cell).replace(",", "").replace("₹", "").replace(" ", "").strip()
        if not cleaned or cleaned in ["-", "–", "N/A", "nil", ""]:
            return 0.0
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


# ─────────────────────────────────────────────
#  HDFC Parser
# ─────────────────────────────────────────────

def _parse_hdfc(lines: list, tables: list) -> pd.DataFrame:
    """Parse HDFC — tries smart table parser first, then line-by-line."""
    transactions = _parse_tables_smart(tables)
    if not transactions:
        print("[HDFC] Table parsing got nothing, trying line-by-line...")
        transactions = _parse_hdfc_lines(lines)
    print(f"[HDFC] Total transactions: {len(transactions)}")
    return _to_dataframe(transactions)


def _parse_hdfc_lines(lines: list) -> list:
    """Line-by-line HDFC fallback."""
    transactions = []
    date_re   = re.compile(r"^\d{2}/\d{2}/\d{2,4}")
    amount_re = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")

    for line in lines:
        line = line.strip()
        if not date_re.match(line):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        date    = parts[0]
        amounts = amount_re.findall(line)
        if not amounts:
            continue
        for raw in amounts:
            amt = float(raw.replace(",", ""))
            if amt > 1:
                narr_end  = line.find(amounts[0])
                narration = re.sub(r"\s+", " ", line[len(date):narr_end]).strip()
                transactions.append([date, narration or "Unknown", amt])
                break

    return transactions


# ─────────────────────────────────────────────
#  SBI Parser
# ─────────────────────────────────────────────

def _parse_sbi(lines: list, tables: list) -> pd.DataFrame:
    """Parse SBI — tries smart table parser first, then line-by-line."""
    transactions = _parse_tables_smart(tables)
    if not transactions:
        print("[SBI] Table parsing got nothing, trying line-by-line...")
        transactions = _parse_sbi_lines(lines)
    print(f"[SBI] Total transactions: {len(transactions)}")
    return _to_dataframe(transactions)


def _parse_sbi_lines(lines: list) -> list:
    """Line-by-line SBI fallback."""
    transactions = []
    date_re   = re.compile(r"(\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}|\d{2}/\d{2}/\d{4})", re.IGNORECASE)
    amount_re = re.compile(r"(\d{1,3}(?:,\d{3})*\.\d{2})")

    for line in lines:
        line = line.strip()
        date_match = date_re.search(line)
        if not date_match:
            continue
        amounts = amount_re.findall(line)
        if not amounts:
            continue
        amt = float(amounts[0].replace(",", ""))
        if amt <= 0:
            continue
        date_end  = date_match.end()
        amt_start = line.find(amounts[0])
        narration = re.sub(r"\s+", " ", line[date_end:amt_start]).strip()
        transactions.append([date_match.group(0), narration or "Unknown", amt])

    return transactions


# ─────────────────────────────────────────────
#  Generic Fallback Parser
# ─────────────────────────────────────────────

def _parse_generic(lines: list) -> pd.DataFrame:
    """Generic parser — tries smart table extraction then text fallback."""
    # Note: tables not passed here — called from parse_bank_statement which already tried tables
    transactions = []
    date_re   = re.compile(
        r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"
        r"|\d{4}[\/\-]\d{2}[\/\-]\d{2}"
        r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{0,4})",
        re.IGNORECASE,
    )
    amount_re = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)")

    for line in lines:
        line = line.strip()
        date_match = date_re.search(line)
        if not date_match:
            continue
        remainder = line[date_match.end():].strip()
        amounts   = amount_re.findall(remainder)
        if not amounts:
            continue
        raw_amt = amounts[-1].replace(",", "")
        try:
            amt = float(raw_amt)
        except ValueError:
            continue
        if amt <= 0:
            continue
        last_pos  = remainder.rfind(amounts[-1])
        narration = re.sub(r"[^\w\s\-\/\(\)@\.]", " ", remainder[:last_pos]).strip()
        transactions.append([date_match.group(0).strip(), narration or "Unknown", amt])

    print(f"[GENERIC] Transactions extracted: {len(transactions)}")
    return _to_dataframe(transactions)


# ─────────────────────────────────────────────
#  UPI Narration Cleaner
# ─────────────────────────────────────────────

# Maps common UPI handles / keywords → readable merchant names
UPI_MERCHANT_MAP = {
    "swiggy":       "Swiggy",
    "zomato":       "Zomato",
    "uber":         "Uber",
    "ola":          "Ola Cabs",
    "netflix":      "Netflix",
    "spotify":      "Spotify",
    "amazon":       "Amazon",
    "flipkart":     "Flipkart",
    "myntra":       "Myntra",
    "bigbasket":    "BigBasket",
    "blinkit":      "Blinkit",
    "dunzo":        "Dunzo",
    "airtel":       "Airtel",
    "jio":          "Jio",
    "hotstar":      "Disney+ Hotstar",
    "phonepe":      "PhonePe",
    "paytm":        "Paytm",
    "gpay":         "Google Pay",
    "irctc":        "IRCTC",
    "makemytrip":   "MakeMyTrip",
    "goibibo":      "Goibibo",
    "bookmyshow":   "BookMyShow",
    "apollo":       "Apollo Pharmacy",
    "1mg":          "1mg",
    "medplus":      "MedPlus",
}

# Patterns that indicate a transaction is a credit (money IN) — skip for expense tracking
CREDIT_PATTERNS = [
    r"salary", r"credited", r"neft cr", r"upi cr", r"interest",
    r"refund", r"cashback", r"reversal", r"/cr/",
]


def _clean_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw bank narrations into human-readable descriptions.
    Handles UPI IDs, NEFT codes, POS entries, etc.
    """
    df = df.copy()
    df["Description"] = df["Description"].apply(_clean_single_narration)

    # Remove credit transactions (money coming IN)
    credit_re = re.compile("|".join(CREDIT_PATTERNS), re.IGNORECASE)
    df = df[~df["Description"].apply(lambda x: bool(credit_re.search(str(x))))]

    return df


def _clean_single_narration(raw: str) -> str:
    """Clean one narration string."""
    if not raw or str(raw).strip() in ["", "nan", "None"]:
        return "Unknown"

    text = str(raw).strip()

    # ── Try to extract merchant from UPI ID ──────────────────────────────────
    # Pattern: UPI/P2M/merchantname@bank or UPI-merchantname-desc@bank
    upi_match = re.search(
        r"UPI[\/\-](?:P2M|P2P|CR|DR)?[\/\-]?([A-Za-z0-9\.\-]+)[@\/]",
        text, re.IGNORECASE
    )
    if upi_match:
        merchant_raw = upi_match.group(1).lower()
        # Check our merchant map
        for key, name in UPI_MERCHANT_MAP.items():
            if key in merchant_raw:
                return name
        # Return cleaned merchant name if not in map
        cleaned = re.sub(r"[^a-zA-Z\s]", " ", merchant_raw).strip().title()
        if len(cleaned) > 2:
            return f"UPI: {cleaned}"

    # ── Check merchant map on full text ──────────────────────────────────────
    text_lower = text.lower()
    for key, name in UPI_MERCHANT_MAP.items():
        if key in text_lower:
            return name

    # ── Clean up common bank prefixes ────────────────────────────────────────
    prefixes_to_remove = [
        r"^TO TRANSFER[-\s]*",
        r"^BY TRANSFER[-\s]*",
        r"^POS\s+",
        r"^NEFT[-\s]*",
        r"^IMPS[-\s]*",
        r"^UPI[-\/\s]*",
        r"^ATM\s+WDL\s*",
        r"^INB\s+",
        r"^MB\s+",
    ]
    for prefix in prefixes_to_remove:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE).strip()

    # Remove reference numbers (long digit strings)
    text = re.sub(r"\b\d{8,}\b", "", text).strip()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text if len(text) > 2 else "Unknown"


# ─────────────────────────────────────────────
#  Flag unknowns for manual review
# ─────────────────────────────────────────────

UNKNOWN_PATTERNS = [
    r"^unknown$",
    r"^upi:\s*$",
    r"^\d+$",                    # pure numbers
    r"^[a-z0-9]{6,}$",          # random alphanumeric ref codes
    r"neft",
    r"imps",
    r"^atm",
    r"transfer",
]


def _flag_unknown_narrations(df: pd.DataFrame) -> pd.DataFrame:
    """Add Needs_Review = True for transactions with unclear descriptions."""
    df = df.copy()
    unknown_re = re.compile("|".join(UNKNOWN_PATTERNS), re.IGNORECASE)
    df["Needs_Review"] = df["Description"].apply(
        lambda x: bool(unknown_re.search(str(x).strip()))
    )
    return df


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _is_date(text: str) -> bool:
    """Quick check if a string looks like a date."""
    date_re = re.compile(
        r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"
        r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
        re.IGNORECASE,
    )
    return bool(date_re.search(str(text).strip()))


def _extract_amount_from_cells(row: list, start_col: int, end_col: int) -> float:
    """Extract the first valid amount from a range of table cells."""
    amount_re = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
    for i in range(start_col, min(end_col + 1, len(row))):
        cell = str(row[i]).strip() if row[i] else ""
        matches = amount_re.findall(cell)
        for m in matches:
            try:
                val = float(m.replace(",", ""))
                if val > 0:
                    return val
            except ValueError:
                continue
    return None


def _to_dataframe(transactions: list) -> pd.DataFrame:
    """Convert transaction list to DataFrame."""
    if not transactions:
        return pd.DataFrame(columns=["Date", "Description", "Amount"])
    df = pd.DataFrame(transactions, columns=["Date", "Description", "Amount"])
    df["Amount"] = pd.to_numeric(df["Amount"].astype(str).str.replace(",", ""), errors="coerce")
    return df


# ─────────────────────────────────────────────
#  6-Month Sample Data
# ─────────────────────────────────────────────

def load_sample_transactions() -> pd.DataFrame:
    """
    Return 6 months of realistic Indian transaction data (Oct 2024 – Mar 2025).
    Includes recurring subscriptions, anomalies, and varied spending patterns.
    """
    data = {
        "Date": [
            "01 Oct 2024", "03 Oct 2024", "05 Oct 2024", "07 Oct 2024", "09 Oct 2024",
            "10 Oct 2024", "12 Oct 2024", "14 Oct 2024", "16 Oct 2024", "18 Oct 2024",
            "20 Oct 2024", "22 Oct 2024", "25 Oct 2024", "27 Oct 2024", "29 Oct 2024",
            "01 Nov 2024", "03 Nov 2024", "05 Nov 2024", "07 Nov 2024", "09 Nov 2024",
            "10 Nov 2024", "12 Nov 2024", "14 Nov 2024", "16 Nov 2024", "18 Nov 2024",
            "20 Nov 2024", "22 Nov 2024", "24 Nov 2024", "26 Nov 2024", "28 Nov 2024",
            "01 Dec 2024", "03 Dec 2024", "05 Dec 2024", "07 Dec 2024", "10 Dec 2024",
            "12 Dec 2024", "14 Dec 2024", "16 Dec 2024", "18 Dec 2024", "20 Dec 2024",
            "22 Dec 2024", "24 Dec 2024", "26 Dec 2024", "28 Dec 2024", "30 Dec 2024",
            "01 Jan 2025", "03 Jan 2025", "05 Jan 2025", "07 Jan 2025", "09 Jan 2025",
            "10 Jan 2025", "12 Jan 2025", "14 Jan 2025", "16 Jan 2025", "18 Jan 2025",
            "20 Jan 2025", "22 Jan 2025", "25 Jan 2025", "27 Jan 2025", "29 Jan 2025",
            "01 Feb 2025", "03 Feb 2025", "05 Feb 2025", "07 Feb 2025", "09 Feb 2025",
            "10 Feb 2025", "12 Feb 2025", "14 Feb 2025", "16 Feb 2025", "18 Feb 2025",
            "20 Feb 2025", "22 Feb 2025", "24 Feb 2025", "26 Feb 2025", "28 Feb 2025",
            "01 Mar 2025", "03 Mar 2025", "05 Mar 2025", "07 Mar 2025", "09 Mar 2025",
            "10 Mar 2025", "12 Mar 2025", "14 Mar 2025", "16 Mar 2025", "18 Mar 2025",
            "20 Mar 2025", "22 Mar 2025", "25 Mar 2025", "27 Mar 2025", "29 Mar 2025",
        ],
        "Description": [
            "Netflix Subscription", "Swiggy Order", "Uber Ride", "Amazon Purchase", "Spotify Premium",
            "Zomato Delivery", "Airtel Broadband Bill", "Ola Cab", "BigBasket Grocery", "Swiggy Order",
            "Flipkart Shopping", "HDFC Credit Card Bill", "Swiggy Instamart", "Uber Ride", "Amazon Prime",
            "Netflix Subscription", "Zomato Delivery", "Uber Ride", "Myntra Shopping", "Spotify Premium",
            "Swiggy Order", "Electricity Bill", "Ola Cab", "BigBasket Grocery", "Zomato Delivery",
            "Amazon Purchase", "HDFC Credit Card Bill", "Swiggy Order", "Uber Ride", "Amazon Prime",
            "Netflix Subscription", "Swiggy Order", "Uber Ride", "Amazon Big Sale", "Spotify Premium",
            "Zomato Delivery", "Airtel Broadband Bill", "Flipkart End of Year Sale", "BigBasket Grocery", "Swiggy Order",
            "Myntra New Year Sale", "HDFC Credit Card Bill", "Goa Trip Hotel Booking", "Uber Ride", "Amazon Prime",
            "Netflix Subscription", "Swiggy Order", "Uber Ride", "Amazon Purchase", "Spotify Premium",
            "Zomato Delivery", "Electricity Bill", "Ola Cab", "BigBasket Grocery", "Swiggy Order",
            "Flipkart Shopping", "HDFC Credit Card Bill", "Swiggy Instamart", "Uber Ride", "Amazon Prime",
            "Netflix Subscription", "Zomato Delivery", "Uber Ride", "Amazon Purchase", "Spotify Premium",
            "Swiggy Order", "Airtel Broadband Bill", "Ola Cab", "BigBasket Grocery", "Zomato Delivery",
            "Myntra Shopping", "HDFC Credit Card Bill", "Swiggy Order", "Uber Ride", "Amazon Prime",
            "Netflix Subscription", "Swiggy Order", "Uber Ride", "Amazon Purchase", "Spotify Premium",
            "Zomato Delivery", "Electricity Bill", "Ola Cab", "BigBasket Grocery", "Swiggy Order",
            "Flipkart Shopping", "HDFC Credit Card Bill", "Swiggy Instamart", "Uber Ride", "Amazon Prime",
        ],
        "Amount": [
            649, 380, 220, 1800, 119,
            450, 999, 180, 2100, 520,
            3200, 4800, 650, 200, 179,
            649, 420, 250, 4500, 119,
            480, 1350, 210, 1950, 390,
            2200, 5100, 580, 230, 179,
            649, 550, 280, 6800, 119,
            620, 999, 5500, 2400, 680,
            4200, 6200, 12500, 310, 179,
            649, 350, 190, 1500, 119,
            410, 1400, 160, 1800, 490,
            2800, 4600, 720, 175, 179,
            649, 440, 230, 2100, 119,
            460, 999, 195, 2050, 370,
            3100, 5000, 540, 210, 179,
            649, 410, 205, 1650, 119,
            395, 1450, 170, 1900, 510,
            2600, 4900, 680, 185, 179,
        ],
    }

    df = pd.DataFrame(data)
    df["Needs_Review"] = False
    return df
"""Message → EvidenceFact. Untrusted text never changes system rules."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from ai.client import ModelClient, ModelError, ModelUnavailableError
from ai.schemas import (
    MESSAGE_EXTRACTION_PROMPT_VERSION,
    MESSAGE_RESPONSE_SCHEMA,
    MESSAGE_SYSTEM_PROMPT,
)
from data.models import Currency, FinancialEvent, FinanceRequest, Message
from evidence.cache import EvidenceCache, cache_key
from evidence.models import (
    EvidenceConfidence,
    EvidenceFact,
    EvidenceSourceType,
    ExtractionMethod,
    FactStatus,
    FactType,
)
from usage.tracker import UsageTracker

PARSER_VERSION = "v1"

_AMOUNT = re.compile(
    r"\b(?P<currency>IDR|EUR|ZAR|INR|USD)\s*(?P<amount>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_PERCENT = re.compile(
    r"(?:increases monthly rent by|menaikkan (?:biaya )?sewa bulanan sebesar)\s*"
    r"(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_LONG_DATE = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)
_INSTRUCTION = re.compile(
    r"(ignore (?:all )?(?:previous|prior) (?:instructions|rules)|"
    r"you must (?:recommend|mark).{0,40}afford|"
    r"override (?:the )?(?:system|rules)|"
    r"system prompt)",
    re.IGNORECASE,
)


def _money(text: str) -> list[tuple[Currency, Decimal]]:
    found: list[tuple[Currency, Decimal]] = []
    for match in _AMOUNT.finditer(text):
        try:
            found.append(
                (Currency(match.group("currency").upper()), Decimal(match.group("amount")))
            )
        except (InvalidOperation, ValueError):
            continue
    return found


def _dates(text: str) -> list[date]:
    found: list[date] = []
    for match in _DATE.finditer(text):
        try:
            found.append(date.fromisoformat(match.group(1)))
        except ValueError:
            continue
    for match in _LONG_DATE.finditer(text):
        try:
            found.append(
                date(int(match.group(3)), _MONTHS[match.group(2).lower()], int(match.group(1)))
            )
        except ValueError:
            continue
    return found


def _fact(
    message: Message,
    fact_type: FactType,
    *,
    notes: str,
    amount: Decimal | None = None,
    currency: Currency | None = None,
    effective_date: date | None = None,
    status: FactStatus = FactStatus.ACTIVE,
    percent: Decimal | None = None,
    category: str | None = None,
    confidence: EvidenceConfidence = EvidenceConfidence.HIGH,
    method: ExtractionMethod = ExtractionMethod.DETERMINISTIC,
    related_event_id: str | None = None,
) -> EvidenceFact:
    return EvidenceFact(
        source_type=EvidenceSourceType.MESSAGE,
        source_id=message.message_id,
        user_id=message.user_id,
        request_id=message.request_id,
        related_event_id=related_event_id if related_event_id is not None else message.related_event_id,
        fact_type=fact_type,
        amount=amount,
        currency=currency,
        effective_date=effective_date,
        status=status,
        supersedes_event_id=None,
        confidence=confidence,
        extraction_method=method,
        raw_reference=message.message_id,
        notes=notes,
        percent=percent,
        category=category,
    )


def _contains(text: str, *phrases: str) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def extract_deterministic(message: Message) -> list[EvidenceFact]:
    text = message.message_text
    # Embedded instructions are data. They never change extraction rules.
    _ = bool(_INSTRUCTION.search(text))
    amounts = _money(text)
    dates = _dates(text)
    facts: list[EvidenceFact] = []

    if _contains(
        text,
        "pay the release charge",
        "pay the processing charge",
        "biaya pencairan",
        "biaya pemrosesan sekarang",
        "you've been selected for a cash prize",
        "anda terpilih untuk menerima hadiah",
    ):
        return [
            _fact(
                message,
                FactType.SCAM_OR_UNTRUSTED_PAYMENT_REQUEST,
                status=FactStatus.UNTRUSTED,
                notes="prize/release-fee request treated as untrusted data",
            )
        ]

    if _contains(
        text,
        "monthly salary has increased to",
        "gaji bulanan anda naik menjadi",
        "gaji bulanan Anda naik menjadi",
    ) and amounts and dates:
        currency, amount = amounts[0]
        facts.append(
            _fact(
                message,
                FactType.SALARY_AMOUNT_CHANGE,
                amount=amount,
                currency=currency,
                effective_date=dates[0],
                category="salary",
                notes="salary increase with effective date",
            )
        )
        return facts

    if _contains(
        text,
        "temporary monthly pay is",
        "gaji bulanan sementara anda",
        "gaji bulanan sementara Anda",
        "jumlah yang lebih rendah masih berlaku",
    ) and amounts:
        currency, amount = amounts[0]
        facts.append(
            _fact(
                message,
                FactType.TEMPORARY_SALARY_CHANGE,
                amount=amount,
                currency=currency,
                effective_date=dates[0] if dates else None,
                category="salary",
                notes="temporary reduced salary for the next payroll",
            )
        )
        return facts

    if _contains(text, "unpaid leave", "cuti tidak dibayar") and amounts:
        currency, amount = amounts[0]
        facts.append(
            _fact(
                message,
                FactType.TEMPORARY_SALARY_CHANGE,
                amount=amount,
                currency=currency,
                category="salary",
                notes="salary reduced for approved unpaid leave",
            )
        )
        return facts

    if _contains(
        text,
        "employment has ended",
        "hubungan kerja anda telah berakhir",
        "hubungan kerja Anda telah berakhir",
        "no regular salary payments scheduled after the final settlement",
        "tidak ada pembayaran gaji rutin",
    ):
        facts.append(
            _fact(
                message,
                FactType.EMPLOYMENT_ENDED,
                category="salary",
                notes="employment ended; do not project regular salary",
            )
        )
        return facts

    if _contains(
        text,
        "seasonal contract has ended",
        "kontrak musiman saat ini telah berakhir",
        "no off-season income",
        "belum ada pendapatan di luar musim",
    ):
        facts.append(
            _fact(
                message,
                FactType.EMPLOYMENT_ENDED,
                category="salary",
                notes="seasonal contract ended; no confirmed future salary",
            )
        )
        return facts

    if _contains(
        text,
        "one household employment record has ended",
        "salah satu sumber pendapatan kerja rumah tangga telah berakhir",
        "remaining confirmed monthly salary",
        "sisa gaji bulanan yang dikonfirmasi",
    ) and amounts:
        currency, amount = amounts[0]
        facts.append(
            _fact(
                message,
                FactType.EMPLOYMENT_ENDED,
                notes="one household employment record ended",
                category="salary",
            )
        )
        facts.append(
            _fact(
                message,
                FactType.SALARY_AMOUNT_CHANGE,
                amount=amount,
                currency=currency,
                category="salary",
                notes="remaining confirmed monthly salary after a role ended",
            )
        )
        return facts

    if _contains(
        text,
        "confirmed salary is now expected on",
        "kini diperkirakan masuk pada",
        "replaces the payroll date",
        "menggantikan tanggal penggajian",
    ) and dates:
        facts.append(
            _fact(
                message,
                FactType.SALARY_PAYMENT_DATE_CHANGE,
                effective_date=dates[0],
                category="salary",
                notes="payday moved; use the revised salary date",
            )
        )
        return facts

    if _contains(
        text,
        "first salary will be",
        "first salary from the new employer",
        "gaji pertama dari perusahaan baru",
        "first salary of",
        "gaji pertama anda sebesar",
        "gaji pertama Anda sebesar",
        "gaji pertama anda akan",
        "your first salary of",
    ) and amounts and dates:
        currency, amount = amounts[0]
        facts.append(
            _fact(
                message,
                FactType.CONFIRMED_FUTURE_INCOME,
                amount=amount,
                currency=currency,
                effective_date=dates[0],
                category="salary",
                notes="confirmed first/new-job salary",
            )
        )
        facts.append(
            _fact(
                message,
                FactType.SALARY_AMOUNT_CHANGE,
                amount=amount,
                currency=currency,
                effective_date=dates[0],
                category="salary",
                notes="new confirmed salary amount from first payday",
            )
        )
        return facts

    if _contains(
        text,
        "salary of",
        "gaji sebesar",
        "is confirmed for",
        "dikonfirmasi untuk",
    ) and amounts and dates and _contains(text, "receiving bank", "bank penerima", "settlement-date", "tanggal penyelesaian"):
        currency, amount = amounts[0]
        facts.append(
            _fact(
                message,
                FactType.CONFIRMED_FUTURE_INCOME,
                amount=amount,
                currency=currency,
                effective_date=dates[0],
                category="salary",
                notes="FX salary confirmed; convert on settlement date",
            )
        )
        return facts

    if _contains(
        text,
        "regular salary of",
        "regular salary resumes",
        "childcare",
        "pembayaran pengasuhan",
        "recurring childcare",
    ) and amounts and dates:
        currency, amount = amounts[0]
        facts.append(
            _fact(
                message,
                FactType.SALARY_AMOUNT_CHANGE,
                amount=amount,
                currency=currency,
                effective_date=dates[0],
                category="salary",
                notes="regular salary resumes on stated date",
            )
        )
        facts.append(
            _fact(
                message,
                FactType.RECURRING_EXPENSE_ADDED,
                effective_date=dates[0],
                category="childcare",
                notes="new recurring childcare payment; amount not stated",
            )
        )
        return facts

    if _contains(
        text,
        "one-time arrears",
        "penyesuaian tunggakan satu kali",
        "one-off adjustment",
    ) and len(amounts) >= 2:
        sal_cur, sal_amt = amounts[0]
        arr_cur, arr_amt = amounts[1]
        facts.append(
            _fact(
                message,
                FactType.SALARY_AMOUNT_CHANGE,
                amount=sal_amt,
                currency=sal_cur,
                category="salary",
                notes="regular salary for the next payroll",
            )
        )
        facts.append(
            _fact(
                message,
                FactType.CONFIRMED_FUTURE_INCOME,
                amount=arr_amt,
                currency=arr_cur,
                category="salary",
                notes="one-time arrears; do not recur",
            )
        )
        return facts

    if _contains(
        text,
        "bonus",
        "commission",
        "komisi",
    ) and _contains(
        text,
        "not been approved",
        "belum disetujui",
        "pending approval",
        "masih menunggu",
        "still pending approval",
        "final performance review",
    ):
        if amounts:
            currency, amount = amounts[0]
            facts.append(
                _fact(
                    message,
                    FactType.SALARY_AMOUNT_CHANGE,
                    amount=amount,
                    currency=currency,
                    category="salary",
                    notes="confirmed base salary; variable pay is separate",
                )
            )
        facts.append(
            _fact(
                message,
                FactType.UNAPPROVED_INCOME,
                status=FactStatus.UNAPPROVED,
                category="bonus" if "bonus" in text.lower() else "commission",
                notes="bonus/commission is not approved cash",
            )
        )
        return facts

    if _contains(
        text,
        "isn't withdrawable",
        "isn’t withdrawable",
        "belum dapat ditarik",
        "payout is still pending",
        "masih tertunda",
        "weekly earnings",
        "penghasilan mingguan",
    ) and _contains(text, "payout", "pembayaran berikutnya"):
        facts.append(
            _fact(
                message,
                FactType.UNAPPROVED_INCOME,
                status=FactStatus.PENDING,
                notes="gig/app payout is pending and not withdrawable",
            )
        )
        return facts

    if _contains(
        text,
        "approved an invoice payment",
        "menyetujui pembayaran faktur",
        "other submitted invoices are still awaiting",
        "faktur lain yang diajukan masih menunggu",
    ) and amounts and dates:
        currency, amount = amounts[0]
        facts.append(
            _fact(
                message,
                FactType.CONFIRMED_FUTURE_INCOME,
                amount=amount,
                currency=currency,
                effective_date=dates[0],
                notes="one approved invoice; other invoices remain unapproved",
            )
        )
        facts.append(
            _fact(
                message,
                FactType.UNAPPROVED_INCOME,
                status=FactStatus.UNAPPROVED,
                notes="unapproved invoices must not be forecast as cash",
            )
        )
        return facts

    rent_pct = _PERCENT.search(text)
    if rent_pct and _contains(text, "rent", "sewa"):
        facts.append(
            _fact(
                message,
                FactType.RECURRING_EXPENSE_CHANGE,
                percent=Decimal(rent_pct.group(1)),
                category="rent",
                notes="lease renewal increases monthly rent from the next payment",
            )
        )
        return facts

    if _contains(
        text,
        "refund has been initiated but has not reached",
        "pengembalian dana sudah diproses, tetapi belum masuk",
        "foreign-currency refund is still processing",
        "pengembalian dana dalam mata uang asing",
    ):
        facts.append(
            _fact(
                message,
                FactType.REFUND_STILL_PENDING,
                status=FactStatus.PENDING,
                notes="refund is not available cash",
            )
        )
        facts.append(
            _fact(
                message,
                FactType.IGNORE_CREDIT,
                status=FactStatus.PENDING,
                notes="pending refund must not be forecast as an inflow",
            )
        )
        return facts

    if _contains(
        text,
        "no units have been sold",
        "holding has not been sold",
        "belum dijual",
        "no cash proceeds",
        "tidak ada transaksi tunai",
        "displayed market value",
        "nilai investasi yang ditampilkan",
    ):
        facts.append(
            _fact(
                message,
                FactType.IGNORE_CREDIT,
                notes="unrealized investment value is not cash",
            )
        )
        return facts

    if _contains(
        text,
        "prize proceeds have reached your account",
        "hasil hadiah sudah masuk",
        "no further scheduled payments",
        "there won’t be another payment",
        "tidak ada pembayaran lain",
    ):
        facts.append(
            _fact(
                message,
                FactType.OTHER_RELEVANT_FINANCIAL_FACT,
                notes="settled prize is closed; do not invent future prizes",
            )
        )
        return facts

    if _contains(
        text,
        "prize claim has been verified and is still in payment processing",
        "klaim hadiah anda sudah diverifikasi",
        "has not been credited",
        "belum masuk ke rekening anda",
    ):
        facts.append(
            _fact(
                message,
                FactType.UNAPPROVED_INCOME,
                status=FactStatus.PENDING,
                notes="prize is still processing and is not cash",
            )
        )
        return facts

    if _contains(
        text,
        "previous debit attempt failed",
        "debit percobaan sebelumnya gagal",
        "another debit will be attempted",
        "debit akan diulang",
    ):
        facts.append(
            _fact(
                message,
                FactType.PAYMENT_RETRY_CONFIRMED,
                notes="failed debit will be retried; keep the obligation",
            )
        )
        return facts

    if _contains(
        text,
        "extra card charge is still being investigated",
        "tagihan kartu tambahan masih dalam penyelidikan",
        "reversal has not been posted",
        "dana pembalikannya belum tercatat",
    ):
        facts.append(
            _fact(
                message,
                FactType.OTHER_RELEVANT_FINANCIAL_FACT,
                notes="possible duplicate/dispute; no reversal posted",
            )
        )
        return facts

    if _contains(
        text,
        "transfer between your two accounts",
        "transfer antara dua rekening",
    ):
        facts.append(
            _fact(
                message,
                FactType.OTHER_RELEVANT_FINANCIAL_FACT,
                notes="internal transfer; do not treat as new income",
            )
        )
        return facts

    if _contains(
        text,
        "minimum payments due on two separate card accounts",
        "dua rekening kartu yang berbeda",
    ):
        facts.append(
            _fact(
                message,
                FactType.OTHER_RELEVANT_FINANCIAL_FACT,
                notes="two card minimums are separate obligations",
            )
        )
        return facts

    if _contains(
        text,
        "reimbursement for your earlier work expense",
        "penggantian atas biaya kerja",
        "no additional reimbursement",
        "tidak ada penggantian tambahan",
    ):
        facts.append(
            _fact(
                message,
                FactType.OTHER_RELEVANT_FINANCIAL_FACT,
                notes="one-time reimbursement is closed",
            )
        )
        return facts

    if _contains(
        text,
        "investment sale have settled",
        "hasil penjualan investasi anda sudah masuk",
        "sale order is complete",
    ):
        facts.append(
            _fact(
                message,
                FactType.OTHER_RELEVANT_FINANCIAL_FACT,
                notes="investment sale already settled as cash history",
            )
        )
        return facts

    if _contains(
        text,
        "charged in a foreign currency",
        "dikenakan dalam mata uang asing",
        "final home-currency amount",
    ) and not _contains(text, "refund"):
        facts.append(
            _fact(
                message,
                FactType.OTHER_RELEVANT_FINANCIAL_FACT,
                notes="foreign-currency charge settles at the dated rate; do not invent FX",
            )
        )
        return facts

    if _contains(
        text,
        "regular salary for the next payroll is already confirmed",
        "gaji rutin untuk penggajian berikutnya sudah dikonfirmasi",
    ):
        facts.append(
            _fact(
                message,
                FactType.CONFIRMED_FUTURE_INCOME,
                category="salary",
                notes="next regular payroll is confirmed; amount not restated",
            )
        )
        return facts

    if _contains(text, "receipt has the final", "the receipt contains the final") and (
        _contains(text, "salary credit") or _contains(text, "employer has confirmed")
    ):
        facts.append(
            _fact(
                message,
                FactType.OTHER_RELEVANT_FINANCIAL_FACT,
                notes="mixed receipt; image holds the charge amount",
                related_event_id=message.related_event_id,
            )
        )
        if amounts:
            currency, amount = amounts[0]
            salary_dates = [item for item in dates if item.day == 15] or dates
            facts.append(
                _fact(
                    message,
                    FactType.CONFIRMED_FUTURE_INCOME,
                    amount=amount,
                    currency=currency,
                    effective_date=salary_dates[-1] if salary_dates else None,
                    category="salary",
                    notes="salary credit confirmed in a mixed receipt+payroll message",
                )
            )
        return facts

    if _contains(text, "receipt has the final", "the receipt contains the final"):
        facts.append(
            _fact(
                message,
                FactType.OTHER_RELEVANT_FINANCIAL_FACT,
                notes="receipt confirms a past payment; image/event holds the amount",
            )
        )
        return facts

    return facts


def needs_llm(message: Message, facts: list[EvidenceFact]) -> bool:
    if facts:
        return False
    text = message.message_text.lower()
    financially_relevant = any(
        token in text
        for token in (
            "salary",
            "gaji",
            "payroll",
            "refund",
            "rent",
            "sewa",
            "bonus",
            "invoice",
            "payout",
            "employment",
        )
    )
    return financially_relevant


def _facts_from_model_payload(
    message: Message,
    payload: dict,
    related_event: FinancialEvent | None,
) -> list[EvidenceFact]:
    allowed_event = related_event.event_id if related_event is not None else message.related_event_id
    facts: list[EvidenceFact] = []
    raw_facts = payload.get("facts")
    if not isinstance(raw_facts, list):
        raise ValueError("facts must be a list")
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        fact_type = FactType(item["fact_type"])
        related = item.get("related_event_id") or None
        if related and allowed_event and related != allowed_event:
            raise ValueError("model invented a related_event_id")
        amount = item.get("amount")
        percent = item.get("percent")
        currency_raw = item.get("currency")
        date_raw = item.get("effective_date")
        confidence = EvidenceConfidence(item.get("confidence", "medium"))
        status = FactStatus(item.get("status", "active"))
        parsed_amount = Decimal(str(amount)) if amount not in (None, "") else None
        if parsed_amount is not None and parsed_amount < 0:
            raise ValueError("amount must be non-negative")
        facts.append(
            _fact(
                message,
                fact_type,
                notes=str(item.get("notes") or "llm extraction"),
                amount=parsed_amount,
                currency=Currency(currency_raw) if currency_raw else None,
                effective_date=date.fromisoformat(date_raw) if date_raw else None,
                status=status,
                percent=Decimal(str(percent)) if percent not in (None, "") else None,
                category=item.get("category"),
                confidence=confidence,
                method=ExtractionMethod.LLM,
                related_event_id=related or allowed_event,
            )
        )
    return facts


def parse_message(
    message: Message,
    *,
    related_event: FinancialEvent | None = None,
    request: FinanceRequest | None = None,
    client: ModelClient | None = None,
    cache: EvidenceCache | None = None,
    tracker: UsageTracker | None = None,
) -> tuple[EvidenceFact, ...]:
    deterministic = extract_deterministic(message)
    if deterministic and not needs_llm(message, deterministic):
        return tuple(deterministic)

    model_name = "none"
    if client is not None:
        model_name = client.text_model
    key = cache_key(
        source_id=message.message_id,
        kind="message",
        version=f"{PARSER_VERSION}:{MESSAGE_EXTRACTION_PROMPT_VERSION}",
        model=model_name,
    )
    if cache is not None:
        cached = cache.get_facts(key)
        if cached is not None:
            if tracker is not None:
                tracker.record(
                    provider="cache",
                    model=model_name,
                    purpose="message_extract",
                    request_id=message.request_id or (request.request_id if request else None),
                    user_id=message.user_id,
                    source_id=message.message_id,
                    prompt_version=MESSAGE_EXTRACTION_PROMPT_VERSION,
                    input_tokens=0,
                    output_tokens=0,
                    cache_hit=True,
                    success=True,
                )
            return cached

    if not needs_llm(message, deterministic):
        return tuple(deterministic)
    if client is None or not client.available():
        return tuple(deterministic)

    user_bits = [
        f"source_type={message.source_type.value}",
        f"message_id={message.message_id}",
        f"message_text={message.message_text}",
    ]
    if related_event is not None:
        user_bits.append(
            "related_event="
            f"{related_event.event_id} type={related_event.event_type.value} "
            f"category={related_event.category} desc={related_event.description}"
        )
    if request is not None:
        user_bits.append(
            f"request_id={request.request_id} request_date={request.request_date.isoformat()}"
        )
    try:
        response = client.complete_json(
            system_prompt=MESSAGE_SYSTEM_PROMPT,
            user_text="\n".join(user_bits),
            purpose="message_extract",
            prompt_version=MESSAGE_EXTRACTION_PROMPT_VERSION,
            request_id=message.request_id or (request.request_id if request else None),
            user_id=message.user_id,
            source_id=message.message_id,
            schema=MESSAGE_RESPONSE_SCHEMA,
        )
        facts = _facts_from_model_payload(message, response.payload, related_event)
    except (ModelUnavailableError, ModelError, ValueError, KeyError):
        return tuple(deterministic)
    if cache is not None:
        cache.put_facts(key, tuple(facts))
    return tuple(facts)

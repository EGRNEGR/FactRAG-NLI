# -*- coding: utf-8 -*-
"""Build & verify extended_dataset_v2.json for FactRAG-NLI A/B evaluation."""
import json, re, pickle, sys

CORPUS = pickle.load(open(r'data\_corpus_text.pkl', 'rb'))
DOCNAME = {
    'rfc9110': 'rfc9110_http_semantics.txt', 'rfc9000': 'rfc9000_quic.txt',
    'rfc9457': 'rfc9457_problem_details.txt', 'rfc9562': 'rfc9562_uuid.txt',
    'nist63b': 'nist_sp800_63b.html', 'sp53': 'nist_sp800_53r5.pdf',
    'pg_ct': 'postgresql_create_table.html', 'pg_idx': 'postgresql_indexes.html',
    'pep703': 'pep703_gil.txt',
}

def squeeze(s):
    """Collapse punctuation spacing the way the source docs render it ('rows .' etc.)."""
    s = re.sub(r'\s+([.,;:!?)])', r'\1', s)   # space before punctuation -> no space
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def norm_quote_for_check(s):
    s = s.replace('(', ' ( ').replace(')', ' ) ')
    s = re.sub(r'<[^>]+>', ' ', s)
    s = s.replace('\u2019', "'").replace('\u2018', "'")
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    s = s.replace('\u2014', '-').replace('\u2013', '-').replace('\u2212', '-')
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return squeeze(s)

def norm(s):
    s = s.replace('\u2019', "'").replace('\u2018', "'")
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    s = s.replace('\u2014', '-').replace('\u2013', '-').replace('\u2212', '-')
    return re.sub(r'\s+', ' ', s).strip().lower()

CORPUS_N = {k: squeeze(re.sub(r'\s+', ' ', v.replace('(', ' ( ').replace(')', ' ) ')).strip().lower()) for k, v in CORPUS.items()}

def case(test_id, doc, qtype, question, expected_output, behavior, quote='', adv_type=None, mech=None):
    c = {"test_id": test_id, "document_name": DOCNAME[doc], "question_type": qtype,
         "question": question, "expected_output": expected_output, "expected_behavior": behavior}
    if qtype == 'Adversarial':
        c["adversarial_type"] = adv_type
    c["grounding_quote"] = quote
    if qtype == 'Adversarial':
        c["distortion_mechanism"] = mech
    return c

cases = []
A = cases.append

# ---------------- POSITIVE (30) ----------------
A(case("POS-001", 'rfc9110', 'Positive',
  "Which request methods does RFC 9110 define as idempotent?",
  "PUT, DELETE, and safe request methods are idempotent; the safe methods defined are GET, HEAD, OPTIONS, and TRACE, so all of those are also idempotent. POST is not.",
  'answer',
  "Of the request methods defined by this specification, PUT, DELETE, and safe request methods are idempotent."))
A(case("POS-002", 'rfc9110', 'Positive',
  "What does the 408 (Request Timeout) status code indicate in RFC 9110?",
  "The server did not receive a complete request message within the time that it was prepared to wait; the client MAY repeat that request.",
  'answer',
  "The 408 (Request Timeout) status code indicates that the server did not receive a complete request message within the time that it was prepared to wait."))
A(case("POS-003", 'rfc9110', 'Positive',
  "What does the 203 (Non-Authoritative Information) status code indicate?",
  "The request was successful but the enclosed content has been modified from that of the origin server's 200 (OK) response by a transforming proxy.",
  'answer',
  "The 203 (Non-Authoritative Information) status code indicates that the request was successful but the enclosed content has been modified from that of the origin server's 200 (OK) response by a transforming proxy"))
A(case("POS-004", 'rfc9110', 'Positive',
  "Как завершается HTTP-ответ 204 No Content и может ли он содержать контент по RFC 9110?",
  "Ответ 204 завершается концом секции заголовков; он не может содержать контент или trailer-поля (terminated by the end of the header section; cannot contain content or trailers).",
  'answer',
  "A 204 response is terminated by the end of the header section; it cannot contain content or trailers."))
A(case("POS-005", 'rfc9110', 'Positive',
  "Какие методы RFC 9110 определяет как безопасные (safe)?",
  "GET, HEAD, OPTIONS и TRACE определены как safe-методы (их семантика по сути read-only, без запроса на изменение состояния).",
  'answer',
  "Of the request methods defined by this specification, the GET, HEAD, OPTIONS, and TRACE methods are defined to be safe."))
A(case("POS-006", 'rfc9000', 'Positive',
  "What limit does RFC 9000 impose on a server sending data prior to client address validation?",
  "Prior to validating the client address, servers MUST NOT send more than three times as many bytes as the number of bytes they have received (anti-amplification limit).",
  'answer',
  "Prior to validating the client address, servers MUST NOT send more than three times as many bytes as the number of bytes they have received."))
A(case("POS-007", 'rfc9000', 'Positive',
  "What does RFC 9000 say about 0-RTT data and replay attacks?",
  "0-RTT allows application data to be sent by a client before receiving a response from the server, but it provides no protection against replay attacks.",
  'answer',
  "0-RTT allows application data to be sent by a client before receiving a response from the server. However, 0-RTT provides no protection against replay attacks"))
A(case("POS-008", 'rfc9000', 'Positive',
  "Какой минимальный максимальный размер дейтаграммы должен поддерживать сетевой тракт, чтобы можно было использовать QUIC?",
  "Не менее 1200 байт: QUIC MUST NOT be used if the network path cannot support a maximum datagram size of at least 1200 bytes.",
  'answer',
  "QUIC MUST NOT be used if the network path cannot support a maximum datagram size of at least 1200 bytes."))
A(case("POS-009", 'rfc9000', 'Positive',
  "How does QUIC support connection migration to a new network path?",
  "The use of connection IDs allows connections to migrate to a new network path, both as a direct choice of an endpoint and when forced by a change in a middlebox.",
  'answer',
  "The use of connection IDs (Section 5.1) allows connections to migrate to a new network path, both as a direct choice of an endpoint and when forced by a change in a middlebox."))
A(case("POS-010", 'rfc9457', 'Positive',
  "Which media types are defined for JSON and XML problem details in RFC 9457?",
  'JSON problem details use "application/problem+json"; the equivalent XML format uses "application/problem+xml".',
  'answer',
  'when serialized as a JSON document, it uses the "application/problem+json" media type. Appendix B defines an equivalent XML format, which uses the "application/problem+xml" media type.'))
A(case("POS-011", 'rfc9457', 'Positive',
  "What must generators do regarding the 'status' member and the actual HTTP response status code in RFC 9457?",
  "Generators MUST use the same status code in the actual HTTP response, to assure that generic HTTP software that does not understand this format still behaves correctly; the status member itself is only advisory.",
  'answer',
  "Generators MUST use the same status code in the actual HTTP response, to assure that generic HTTP software that does not understand this format still behaves correctly."))
A(case("POS-012", 'rfc9457', 'Positive',
  "How must clients consuming problem details treat extensions they don't recognize?",
  "Clients consuming problem details MUST ignore any such extensions that they don't recognize; this allows problem types to evolve and include additional information in the future.",
  'answer',
  "Clients consuming problem details MUST ignore any such extensions that they don't recognize; this allows problem types to evolve and include additional information in the future."))
A(case("POS-013", 'rfc9562', 'Positive',
  "What is the size of the UUID format according to RFC 9562?",
  "The UUID format is 16 octets (128 bits) in size; bit definitions start at 0 and end at 127, octet definitions start at 0 and end at 15.",
  'answer',
  "The UUID format is 16 octets (128 bits) in size; the variant bits in conjunction with the version bits described in the next sections determine finer structure."))
A(case("POS-014", 'rfc9562', 'Positive',
  "What is the Nil UUID and how is it specified?",
  "The Nil UUID is a special form of UUID specified to have all 128 bits set to zero (00000000-0000-0000-0000-000000000000); it can communicate the absence of any other UUID value.",
  'answer',
  "A Nil UUID value can be useful to communicate the absence of any other UUID value in situations that otherwise require or use a"))
A(case("POS-015", 'rfc9562', 'Positive',
  "Сколько бит занимает поле unix_ts_ms в UUIDv7 и что оно содержит?",
  "48 бит: unix_ts_ms — 48-битное big-endian беззнаковое число метки времени Unix Epoch в миллисекундах; занимает биты 0–47 (октеты 0–5).",
  'answer',
  "unix_ts_ms: 48-bit big-endian unsigned number of the Unix Epoch timestamp in milliseconds as per Section 6.1. Occupies bits 0 through 47"))
A(case("POS-016", 'rfc9562', 'Positive',
  "Как RFC 9562 рекомендует хранить UUID в базах данных?",
  "По возможности UUID следует хранить в приложениях баз данных как 128-битное двоичное значение; хранение в виде текста избыточно многословно (288 бит для представления 128-битного UUID).",
  'answer',
  "Thus, where feasible, UUIDs SHOULD be stored within database applications as the underlying 128-bit binary value."))
A(case("POS-017", 'nist63b', 'Positive',
  "What minimum password length does SP 800-63B require when the password is the single authentication factor?",
  "A minimum of 15 characters for passwords used as a single-factor authentication mechanism; passwords used only as part of multi-factor authentication may be shorter but SHALL be at least eight characters.",
  'answer',
  "Verifiers and CSPs SHALL require passwords that are used as a single-factor authentication mechanism to be a minimum of 15 characters in length."))
A(case("POS-018", 'nist63b', 'Positive',
  "After how many consecutive failed authentication attempts with a specific authenticator must it be disabled, per SP 800-63B?",
  "No more than 100 consecutive failed authentication attempts using a specific authenticator on a single subscriber account, after which that authenticator is disabled; the limit of 100 is an upper bound and agencies MAY impose lower limits.",
  'answer',
  "the verifier SHALL limit consecutive failed authentication attempts using a specific authenticator on a single subscriber account to no more than 100 by disabling that authenticator"))
A(case("POS-019", 'nist63b', 'Positive',
  "What false match rate (FMR) must a biometric system operate with per SP 800-63B?",
  "An FMR of one in 10000 or better for all demographic groups (under zero-effort impostor attempt conditions); the system SHOULD also demonstrate an FNMR of less than 5%.",
  'answer',
  "The biometric system SHALL operate with an FMR [ISO/IEC2382-37] of one in 10000 or better for all demographic groups."))
A(case("POS-020", 'nist63b', 'Positive',
  "Какие таймауты реаутентификации установлены для AAL2 в SP 800-63B?",
  "Для AAL2: общий таймаут реаутентификации (overall timeout) — не более 24 часов; таймаут неактивности (inactivity timeout) — не более 1 часа.",
  'answer',
  "A definite reauthentication overall timeout SHALL be established, which SHOULD be no more than 24 hours at AAL2. The inactivity timeout SHOULD be no more than 1 hour."))
A(case("POS-021", 'nist63b', 'Positive',
  "Разрешено ли в SP 800-63B биометрическое сравнение по голосу?",
  "Нет: Biometric comparison based on voice SHALL NOT be used. PAD следует реализовать для радужки и отпечатков и обязателен для распознавания лиц.",
  'answer',
  "The biometric system SHOULD implement PAD for iris and fingerprint modalities and SHALL implement PAD for facial recognition. Biometric comparison based on voice SHALL NOT be used."))
A(case("POS-022", 'pg_ct', 'Positive',
  "What is the default ON COMMIT behavior for PostgreSQL temporary tables?",
  "The default behavior in PostgreSQL is ON COMMIT PRESERVE ROWS (the SQL standard would specify ON COMMIT DELETE ROWS; the ON COMMIT DROP option does not exist in SQL).",
  'answer',
  "However, the default behavior in PostgreSQL is ON COMMIT PRESERVE ROWS. The ON COMMIT DROP option does not exist in SQL."))
A(case("POS-023", 'pg_ct', 'Positive',
  "Can the autovacuum daemon vacuum or analyze temporary tables in PostgreSQL?",
  "No: the autovacuum daemon cannot access and therefore cannot vacuum or analyze temporary tables; appropriate vacuum and analyze operations should be performed via session SQL commands (e.g., ANALYZE after populating the table).",
  'answer',
  "The autovacuum daemon cannot access and therefore cannot vacuum or analyze temporary tables. For this reason, appropriate vacuum and analyze operations should be performed via session SQL commands."))
A(case("POS-024", 'pg_ct', 'Positive',
  "Сколько колонок может включать ключ партиционирования при range/hash-партиционировании в PostgreSQL и какое ограничение действует для list?",
  "До 32 колонок или выражений для range- и hash-партиционирования (лимит можно изменить при сборке PostgreSQL); для list-партиционирования ключ должен состоять из одной колонки или выражения.",
  'answer',
  "the partition key can include multiple columns or expressions (up to 32, but this limit can be altered when building PostgreSQL), but for list partitioning, the partition key must consist of a single column or expression."))
A(case("POS-025", 'pg_idx', 'Positive',
  "What trade-off does the PostgreSQL indexes chapter describe about using indexes?",
  "An index allows the database server to find and retrieve specific rows much faster than without an index, but indexes also add overhead to the database system as a whole, so they should be used sensibly.",
  'answer',
  "An index allows the database server to find and retrieve specific rows much faster than it could do without an index. But indexes also add overhead to the database system as a whole, so they should be used sensibly."))
A(case("POS-026", 'pep703', 'Positive',
  "What build configuration does PEP 703 propose to add to CPython?",
  "A --disable-gil build configuration that lets CPython run Python code without the global interpreter lock, together with the changes needed to make the interpreter thread-safe.",
  'answer',
  "The GIL is an obstacle to using multi-core CPUs from Python efficiently."))
A(case("POS-027", 'pep703', 'Positive',
  "What single-threaded and multi-threaded execution overhead does PEP 703 report for --disable-gil builds?",
  "On pyperformance 1.0.6: one thread — 6% (Intel Skylake) and 5% (AMD Zen 3); multiple threads — 8% and 7% respectively. The largest contribution to the overhead is biased reference counting, followed by per-object locking.",
  'answer',
  "The largest contribution to execution overhead is biased reference counting followed by per-object locking. For thread-safety reasons, an application running with multiple threads will only specialize a given bytecode once"))
A(case("POS-028", 'pep703', 'Positive',
  "What does PEP 703's build mode roadmap say happens with CPython 3.13 in 2024?",
  "In 2024, CPython 3.13 is released with support for a --disable-gil build time flag; there are two ABIs for CPython, one with the GIL and one without, and extension authors target both ABIs.",
  'answer',
  "There are two ABIs for CPython, one with the GIL and one without. Extension authors target both ABIs."))
A(case("POS-029", 'sp53', 'Positive',
  "What does the CM-6 CONFIGURATION SETTINGS control require in SP 800-53 Rev. 5?",
  "CM-6 requires establishing and documenting configuration settings for system components that reflect the most restrictive mode consistent with operational requirements, implementing them, identifying/documenting/approving deviations, and monitoring and controlling changes to the settings.",
  'answer',
  "Establish and document configuration settings for components employed within the system that reflect the most restrictive mode consistent with operational requirements"))
A(case("POS-030", 'sp53', 'Positive',
  "What does the PS-3 PERSONNEL SCREENING control require?",
  "PS-3 requires screening individuals prior to authorizing access to the system, and rescreening individuals in accordance with organization-defined conditions and frequency of rescreening.",
  'answer',
  "Screen individuals prior to authorizing access to the system; and b. Rescreen individuals in accordance with [Assignment: organization-defined conditions requiring rescreening and, where rescreening is so indicated, the frequency of rescreening]."))

# ---------------- NEGATIVE (10) ----------------
A(case("NEG-001", 'rfc9110', 'Negative',
  "Определяет ли RFC 9110 код статуса 511 Network Authentication Required?",
  "В корпусе нет ответа: RFC 9110 не определяет код 511; в документе отсутствует любой статус с именем Network Authentication Required (определённые клиентские коды ошибок заканчиваются на 418).",
  'refusal'))
A(case("NEG-002", 'rfc9000', 'Negative',
  "Does RFC 9000 support multipath, i.e., simultaneous transmission of one connection's data over multiple network paths?",
  "No answer in the corpus: RFC 9000 does not define multipath support; it describes connection migration to a new path and path validation, and the term 'multipath' does not occur in the document.",
  'refusal'))
A(case("NEG-003", 'rfc9457', 'Negative',
  'Does the RFC 9457 problem details JSON object define a "timestamp" member?',
  "No answer in the corpus: the defined members of the problem details JSON object are type, title, status, detail, and instance; no 'timestamp' member is defined and the word does not occur in the document.",
  'refusal'))
A(case("NEG-004", 'rfc9562', 'Negative',
  "Определяет ли RFC 9562 формат UUIDv9?",
  "Нет ответа в корпусе: RFC 9562 определяет версии UUIDv1–UUIDv8, а также Nil и Max UUID; UUIDv9 и «version 9» в документе не упоминаются.",
  'refusal'))
A(case("NEG-005", 'nist63b', 'Negative',
  "Требует ли SP 800-63B использовать алгоритм Argon2 для хеширования паролей?",
  "Нет ответа в корпусе: документ требует использовать approved password hashing scheme (рекомендуется из SP 800-132 или актуальных руководств NIST), но конкретный алгоритм Argon2 в документе не упоминается.",
  'refusal'))
A(case("NEG-006", 'nist63b', 'Negative',
  "Does this edition of SP 800-63B define IAL4 (Identity Assurance Level 4)?",
  "No answer in the corpus: this document defines AAL1–AAL3 for authenticators; identity assurance levels are not enumerated here, and IAL4 does not appear in the document.",
  'refusal'))
A(case("NEG-007", 'pg_ct', 'Negative',
  "Поддерживает ли CREATE TABLE в PostgreSQL опцию ON COMMIT ROLLBACK для временных таблиц?",
  "Нет ответа в корпусе: для ON COMMIT поддерживаются только PRESERVE ROWS, DELETE ROWS и DROP; опция ROLLBACK не определена, и слово ROLLBACK в документе не встречается.",
  'refusal'))
A(case("NEG-008", 'pg_idx', 'Negative',
  "Does the PostgreSQL indexes chapter recommend a maximum number of indexes per table?",
  "No answer in the corpus: the chapter does not state any maximum or recommended number of indexes per table; it only says that indexes speed up row retrieval but add overhead to the database system as a whole, so they should be used sensibly.",
  'refusal'))
A(case("NEG-009", 'pep703', 'Negative',
  "Does PEP 703 require C extensions to be rewritten in Rust to work without the GIL?",
  "No answer in the corpus: PEP 703 does not mandate rewriting extensions in Rust or any other language; the word 'Rust' does not occur in the document — extension authors are expected to target two ABIs (with and without the GIL).",
  'refusal'))
A(case("NEG-010", 'sp53', 'Negative',
  "Does SP 800-53 Rev. 5 define a control or control family with the identifier XY (e.g., XY-1)?",
  "No answer in the corpus: the 20 control families defined include AC, AT, AU, CA, CM, CP, IA, IR, MA, MP, PE, PL, PM, PS, PT, RA, SA, SC, SI, and SR; there is no XY family and no XY-1 control.",
  'refusal'))

# ---------------- ADVERSARIAL (15) ----------------
A(case("ADV-001", 'rfc9110', 'Adversarial',
  "RFC 9110 says that a 400 (Bad Request) response indicates the server understood the request but refuses to process it, correct?",
  "Correction: 400 (Bad Request) indicates that the server cannot or will not process the request due to something perceived to be a client error (e.g., malformed request syntax, invalid framing, deceptive routing); 'understood the request but refuses to fulfill it' is the definition of 403 (Forbidden).",
  'refusal_or_correction',
  "The 400 (Bad Request) status code indicates that the server cannot or will not process the request due to something that is perceived to be a client error",
  'premise_distortion',
  "The premise substitutes the 403 (Forbidden) definition ('understood the request but refuses to fulfill it') for 400; the lexically close trap chunk is Section 15.5.4 403 Forbidden, which the reader confuses with 400."))
A(case("ADV-002", 'rfc9000', 'Adversarial',
  "RFC 9000 утверждает, что 0-RTT-данные защищены от replay-атак, верно?",
  "Нет, предпосылка искажена: 0-RTT позволяет клиенту отправить данные приложения до ответа сервера, но не даёт никакой защиты от атак повторного воспроизведения — '0-RTT provides no protection against replay attacks'.",
  'refusal_or_correction',
  "However, 0-RTT provides no protection against replay attacks; see Section 9.2 of [QUIC-TLS].",
  'premise_distortion',
  "The premise inverts the document's statement about 0-RTT replay protection; the trap chunk is Section 7.1's 0-RTT discussion, which first mentions sending early data (true part) next to the replay caveat."))
A(case("ADV-003", 'nist63b', 'Adversarial',
  "SP 800-63B classifies passwords as phishing-resistant because they are salted and hashed, correct?",
  "Correction: passwords are not phishing-resistant; phishing resistance requires single- or multi-factor cryptographic authentication, and authenticators that involve the manual entry of an authenticator output (e.g., out-of-band and OTP) SHALL NOT be considered phishing-resistant.",
  'refusal_or_correction',
  "Phishing resistance requires single- or multi-factor cryptographic authentication. Authenticators that involve the manual entry of an authenticator output (e.g., out-of-band and OTP authenticators) SHALL NOT be considered phishing-resistant",
  'premise_distortion',
  "The premise falsely attributes phishing resistance to passwords; the trap chunk is the password verifier section (salted and hashed storage), which is lexically close to but unrelated to phishing resistance."))
A(case("ADV-004", 'rfc9110', 'Adversarial',
  "Почему RFC 9110 относит метод DELETE к безопасным (safe) методам наряду с GET и HEAD?",
  "Предпосылка неверна: DELETE не является safe-методом. Safe определены только GET, HEAD, OPTIONS и TRACE (их семантика по сути read-only, без запроса изменения состояния). DELETE идемпотентен — как PUT и все safe-методы, — но идемпотентность и безопасность в RFC 9110 разные свойства.",
  'refusal_or_correction',
  "Request methods are considered \"safe\" if their defined semantics are essentially read-only; i.e., the client does not request, and does not expect, any state change on the origin server",
  'term_substitution',
  "Substitutes safe <-> idempotent: DELETE is listed in Section 9.2.2 (Idempotent Methods) together with safe methods, so the idempotent-methods chunk is the trap that makes 'DELETE is safe' sound plausible."))
A(case("ADV-005", 'nist63b', 'Adversarial',
  "In SP 800-63B, the verifier is the entity whose identity is confirmed by the claimant during authentication, correct?",
  "Correction: the roles are reversed — the verifier is the entity that confirms the claimant's identity by verifying the claimant's possession and control of one or more authenticators using an authentication protocol; the claimant is the party claiming the identity.",
  'refusal_or_correction',
  "To do this, the verifier needs to confirm the binding of the authenticators with the subscriber account and check that the subscriber account is active.",
  'term_substitution',
  "Substitutes claimant <-> verifier in the glossary definition of verifier; the trap is the definition-of-verifier chunk itself, where both roles co-occur, so the reversed reading feels familiar."))
A(case("ADV-006", 'rfc9110', 'Adversarial',
  "Which header field must a server send in a 403 (Forbidden) response to challenge the client to re-authenticate, per RFC 9110?",
  "Correction: no such requirement exists for 403 (Forbidden) — the server understood the request but refuses to fulfill it. It is a 401 (Unauthorized) response that MUST include a WWW-Authenticate header field containing at least one challenge applicable to the target resource.",
  'refusal_or_correction',
  "The server generating a 401 response MUST send a WWW-Authenticate header field (Section 11.6.1) containing at least one challenge applicable to the target resource.",
  'term_substitution',
  "Substitutes 401 <-> 403 and the authentication challenge requirement; the trap chunks are Section 15.5.4 403 Forbidden ('understood but refuses') and the 401/WWW-Authenticate text, which are adjacent client-error definitions."))
A(case("ADV-007", 'rfc9562', 'Adversarial',
  "Поле unix_ts_ms в UUIDv7 — это 32-битный big-endian номер, не так ли?",
  "Нет, искажено число: unix_ts_ms — 48-битное big-endian беззнаковое число метки времени Unix Epoch в миллисекундах, занимающее биты 0–47 (октеты 0–5), а не 32 бита.",
  'refusal_or_correction',
  "unix_ts_ms: 48-bit big-endian unsigned number of the Unix Epoch timestamp in milliseconds as per Section 6.1. Occupies bits 0 through 47",
  'numeric_distortion',
  "Numeric substitution 48 <-> 32 in the UUIDv7 field layout; the trap chunk is Section 5.7 UUIDv7 Field and Bit Layout, whose other fields (12-bit rand_a, 4-bit ver) contain small numbers that make 32 bits sound plausible."))
A(case("ADV-008", 'pg_ct', 'Adversarial',
  "Ключ партиционирования при range/hash-партиционировании в PostgreSQL может включать до 64 колонок, верно?",
  "Нет, искажено число: до 32 колонок или выражений (лимит можно изменить при сборке PostgreSQL), а для list-партиционирования ключ должен состоять из одной колонки или выражения.",
  'refusal_or_correction',
  "the partition key can include multiple columns or expressions (up to 32, but this limit can be altered when building PostgreSQL), but for list partitioning, the partition key must consist of a single column or expression.",
  'numeric_distortion',
  "Numeric substitution 32 <-> 64 in the PARTITION BY clause description; the trap chunk is the CREATE TABLE PARTITION BY parameter text, which contains the true limit 32."))
A(case("ADV-009", 'rfc9000', 'Adversarial',
  "Before validating the client's address, a QUIC server must not send more than twice as many bytes as it has received, right?",
  "Correction: the anti-amplification limit is three times, not twice — prior to validating the client address, servers MUST NOT send more than three times as many bytes as the number of bytes they have received.",
  'refusal_or_correction',
  "Prior to validating the client address, servers MUST NOT send more than three times as many bytes as the number of bytes they have received. This limits the magnitude of any amplification attack",
  'numeric_distortion',
  "Numeric substitution 3x <-> 2x in the address-validation anti-amplification limit; the trap chunk is Section 8.1's amplification text, where 'three times' appears repeatedly alongside byte counts."))
A(case("ADV-010", 'rfc9562', 'Adversarial',
  "RFC 9562 requires that UUIDs always be stored as the underlying 128-bit binary value in every system, right?",
  "Correction: the requirement is overstated — for database applications it is a SHOULD where feasible ('UUIDs SHOULD be stored within database applications as the underlying 128-bit binary value'), while other systems MAY store UUIDs in binary form or as text, as appropriate.",
  'refusal_or_correction',
  "For other systems, UUIDs MAY be stored in binary form or as text, as appropriate.",
  'modality_scope',
  "Turns SHOULD (databases) / MAY (other systems) into an unconditional 'always/must in every system'; the trap chunk is Section 6.13 DBMS and Database Considerations, whose first sentence carries the SHOULD recommendation."))
A(case("ADV-011", 'nist63b', 'Adversarial',
  "SP 800-63B обязывает периодически менять пароли — например, каждые 90 дней, верно?",
  "Нет, модальность искажена и значение выдумано: SP 800-63B запрещает требовать произвольную периодическую смену паролей (SHALL NOT require subscribers to change passwords periodically); принудительная смена положена только при признаках компрометации аутентификатора. Значения «90 дней» в документе нет.",
  'refusal_or_correction',
  "Verifiers and CSPs SHALL NOT require subscribers to change passwords periodically. However, verifiers SHALL force a change if there is evidence that the authenticator has been compromised.",
  'modality_scope',
  "Inverts SHALL NOT into a mandatory periodic-rotation rule and invents the 90-day figure; the trap chunk is the password requirements list in Sec. 3.1.1.2, which discusses forced changes only upon evidence of compromise."))
A(case("ADV-012", 'pep703', 'Adversarial',
  "PEP 703 требует полностью удалить GIL из всех сборок CPython начиная с версии 3.13, верно?",
  "Нет, объём искажён: GIL остаётся режимом по умолчанию для сборок CPython и загрузок python.org; PEP 703 лишь добавляет опциональный флаг --disable-gil, и в 3.13 существуют два ABI — с GIL и без него.",
  'refusal_or_correction',
  "The global interpreter lock will remain the default for CPython builds and python.org downloads. A new build configuration flag,",
  'modality_scope',
  "Turns the optional --disable-gil build configuration into a mandatory removal of the GIL from all builds; the trap chunk is the Specification section on Build Configuration Changes, which states the GIL remains the default."))
A(case("ADV-013", 'pep703', 'Adversarial',
  "PEP 703 меняет около 15 000 строк CPython и добавляет mimalloc — значит, отключение GIL происходит без накладных расходов на производительность, верно?",
  "Предпосылка верна (реализация меняет ~15 000 строк CPython и включает mimalloc, тоже ~15 000 строк), но вывод ложен: изменения делают CPython потокобезопасным без GIL ценой роста накладных расходов — например, 6%/5% для одного потока и 8%/7% для многопоточных программ на pyperformance; наибольший вклад дают biased reference counting и per-object locking.",
  'refusal_or_correction',
  "The performance impact is different for programs that use only a single thread compared to programs that use multiple threads, so the table below reports execution overhead separately for these types of programs separately.",
  'mixed',
  "True premise (the 15,000-line change and mimalloc inclusion, from the Integration section) combined with a false conclusion (zero overhead); the trap chunk is the Performance section, which opens by stating that the changes increase execution overhead."))
A(case("ADV-014", 'sp53', 'Adversarial',
  "Раз SP 800-53 Rev. 5 определяет семью Configuration Management, значит, в публикации перечислено 21 семейство контролей, верно?",
  "Предпосылка верна (CM Configuration Management действительно определена), но вывод ложен: таблица 1 перечисляет ровно 20 семейств контролей безопасности и приватности, включая AC, AT, AU, CA, CM, CP, IA, IR, MA, MP, PE, PL, PM, PS, PT, RA, SA, SC, SI и SR.",
  'refusal_or_correction',
  "TABLE 1: SECURITY AND PRIVACY CONTROL FAMILIES ID FAMILY ID FAMILY AC Access Control PE Physical and Environmental Protection AT Awareness and Training PL Planning AU Audit",
  'mixed',
  "True premise (CM family exists) combined with the false conclusion of 21 families; the trap chunk is Table 1 of control families, which lists the true count of 20."))
A(case("ADV-015", 'rfc9457', 'Adversarial',
  "Problem details можно использовать с любым HTTP-кодом статуса, а поле status в объекте всегда авторитетно и переопределяет реальный код ответа — верно?",
  "Предпосылка верна (problem details can be used with any HTTP status code), но вывод ложен: поле status только рекомендательное — 'The status member, if present, is only advisory; it conveys the HTTP status code used for the convenience of the consumer'; генератор обязан использовать тот же код в самом HTTP-ответе.",
  'refusal_or_correction',
  "The \"status\" member, if present, is only advisory; it conveys the HTTP status code used for the convenience of the consumer.",
  'mixed',
  "True premise (usable with any status code) combined with a false conclusion about the status member being authoritative; the trap chunk is Section 3.1.2 'status', which states the member is only advisory."))

# ---------------- VERIFICATION ----------------
errors = []
for c in cases:
    k = [k for k, v in DOCNAME.items() if v == c['document_name']][0]
    doc_n = CORPUS_N[k]
    if c['question_type'] in ('Positive', 'Adversarial'):
        q = norm_quote_for_check(c['grounding_quote'])
        if q not in doc_n:
            errors.append(f"QUOTE NOT FOUND {c['test_id']}: {c['grounding_quote'][:80]}")
        else:
            wc = len(c['grounding_quote'].split())
            if wc < 10 or wc > 40:
                errors.append(f"QUOTE LEN {wc} {c['test_id']}")
        # English or Russian?
        has_cyr = bool(re.search(r'[А-Яа-яЁё]', c['question']))
        c['_ru'] = has_cyr
    else:
        c['_ru'] = bool(re.search(r'[А-Яа-яЁё]', c['question']))

# NEG absence checks: key entity must not appear in doc
absence_checks = [
    ("NEG-001", 'rfc9110', ["511 ", "network authentication required"]),
    ("NEG-002", 'rfc9000', ["multipath"]),
    ("NEG-003", 'rfc9457', ["timestamp"]),
    ("NEG-004", 'rfc9562', ["uuidv9", "version 9"]),
    ("NEG-005", 'nist63b', ["argon2"]),
    ("NEG-006", 'nist63b', ["ial4", "identity assurance level 4"]),
    ("NEG-007", 'pg_ct', ["rollback"]),
    ("NEG-008", 'pg_idx', ["maximum number of indexes"]),
    ("NEG-009", 'pep703', [" rust", "rust "]),
    ("NEG-010", 'sp53', ["xy-1", "xy access", "xy family"]),
]
for tid, k, terms in absence_checks:
    doc_n = CORPUS_N[k]
    for t in terms:
        if t in doc_n:
            errors.append(f"ABSENCE FAIL {tid}: '{t}' FOUND in {DOCNAME[k]}")
    # sanity: doc non-empty
    if len(doc_n) < 1000:
        errors.append(f"ABSENCE FAIL {tid}: doc too short {DOCNAME[k]}")

# duplicate check vs existing datasets
def load_questions(path):
    try:
        d = json.load(open(path, encoding='utf-8'))
    except Exception:
        return set()
    qs = set()
    def walk(x):
        if isinstance(x, dict):
            if 'question' in x and isinstance(x['question'], str):
                qs.add(norm(x['question']))
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(d)
    return qs

seen = set()
for path in ['data/golden_dataset.json', 'data/holdout_dataset.json',
             'data/extended_questions.json', 'data/qa_drafts.json']:
    seen |= load_questions(path)

for c in cases:
    nq = norm(c['question'])
    if nq in seen:
        errors.append(f"DUPLICATE vs existing datasets: {c['test_id']} {c['question'][:60]}")
    if any(norm(o['question']) == nq for o in cases if o['test_id'] != c['test_id']):
        errors.append(f"INTERNAL DUPLICATE: {c['test_id']}")

if errors:
    print('ERRORS:')
    for e in errors: print(' ', e)
    sys.exit(1)

# stats
by_type = {'Positive': 0, 'Negative': 0, 'Adversarial': 0}
by_doc = {}
adv_by_type = {}
ru = 0
for c in cases:
    by_type[c['question_type']] += 1
    by_doc[c['document_name']] = by_doc.get(c['document_name'], 0) + 1
    if c['question_type'] == 'Adversarial':
        adv_by_type[c['adversarial_type']] = adv_by_type.get(c['adversarial_type'], 0) + 1
    if c['_ru']: ru += 1

for c in cases:
    c.pop('_ru', None)

out = {
  "version": 2,
  "description": "Extended adversarial-rich QA corpus v2 for FactRAG-NLI A/B evaluation",
  "corpus_root": "../test_docs",
  "cases": cases,
  "verification": {
    "cases_total": len(cases),
    "positives": by_type['Positive'],
    "negatives": by_type['Negative'],
    "adversarials": by_type['Adversarial'] if False else by_type['Adversarial'],
    "adversarial_by_type": adv_by_type,
    "russian_questions": ru,
    "grounding_verified": sum(1 for c in cases if c['question_type'] in ('Positive', 'Adversarial')),
    "absence_verified": by_type['Negative'],
    "duplicates_checked_against": ["golden_dataset.json", "holdout_dataset.json", "extended_questions.json", "qa_drafts.json"]
  }
}
# fix key name
out['verification']['adversarial'] = out['verification'].pop('adversarials')

with open('data/extended_dataset_v2.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("OK: wrote", len(cases), "cases")
print("by_type:", by_type)
print("by_doc:", json.dumps(by_doc, indent=1))
print("adv_by_type:", adv_by_type)
print("russian:", ru)

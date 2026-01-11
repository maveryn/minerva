Retrieval query sanity checks (local engine, global label_type=all)

Notes:
- Runs against the local RetrievalEngine (not the HTTP server).
- Uses global retrieval (`label_type=all`) with `retrieval_mode=per_type` (global docs loaded on demand).
- Exact ID extraction is enabled for all label types.
- Results are capped to the requested topk.
- Detection label docs are omitted from the global corpus, so no detection_id tests are included.
- Global retrieval can return mixed label types; these checks verify that intended IDs still surface near the top.

## attack_tactic_id

Case 1 (global)
query: "privilege escalation tactic"
topk: 4
results:
- attack_tactic_id:TA0004 | Privilege Escalation
- capec_id:CAPEC-233 | Privilege Escalation
- attack_technique_id:T1652 | Device Driver Discovery
- mitigation_id:M1019 | Threat Intelligence Program

Case 2 (global)
query: "TA0004 TA0006 TA0008"
topk: 5
results:
- attack_tactic_id:TA0004 | Privilege Escalation
- attack_tactic_id:TA0008 | Lateral Movement
- attack_tactic_id:TA0006 | Credential Access
- attack_tactic_id:TA0002 | Execution
- attack_tactic_id:TA0040 | Impact

Case 3 (global)
query: "TA0003 persistence"
topk: 6
results:
- attack_tactic_id:TA0003 | Persistence
- attack_technique_id:T1546.018 | Python Startup Hooks
- attack_technique_id:T1070.009 | Clear Persistence
- attack_technique_id:T1547.001 | Registry Run Keys / Startup Folder
- attack_technique_id:T1547.014 | Active Setup
- attack_technique_id:T1112 | Modify Registry

## attack_technique_id

Case 1 (global)
query: "spearphishing attachment malicious file"
topk: 4
results:
- mitigation_id:M1017 | User Training
- attack_technique_id:T1598.002 | Spearphishing Attachment
- attack_technique_id:T1566.001 | Spearphishing Attachment
- mitigation_id:M1049 | Antivirus/Antimalware

Case 2 (global)
query: "T1566 T1566.001 T1566.002"
topk: 5
results:
- attack_technique_id:T1566.002 | Spearphishing Link
- attack_technique_id:T1566.001 | Spearphishing Attachment
- attack_technique_id:T1566 | Phishing
- mitigation_id:M1021 | Restrict Web-Based Content
- mitigation_id:M1017 | User Training

Case 3 (global)
query: "T1059.003 powershell scripting"
topk: 6
results:
- attack_technique_id:T1059.003 | Windows Command Shell
- mitigation_id:M1045 | Code Signing
- mitigation_id:M1049 | Antivirus/Antimalware
- mitigation_id:M1038 | Execution Prevention
- mitigation_id:M1042 | Disable or Remove Feature or Program
- mitigation_id:M1033 | Limit Software Installation

## capec_id

Case 1 (global)
query: "buffer overflow environment variables"
topk: 4
results:
- capec_id:CAPEC-10 | Buffer Overflow via Environment Variables
- capec_id:CAPEC-46 | Overflow Variables and Tags
- cwe_id:CWE-122 | Heap-based Buffer Overflow
- cwe_id:CWE-680 | Integer Overflow to Buffer Overflow

Case 2 (global)
query: "CAPEC-10 CAPEC-100 CAPEC-108"
topk: 5
results:
- capec_id:CAPEC-100 | Overflow Buffers
- capec_id:CAPEC-10 | Buffer Overflow via Environment Variables
- capec_id:CAPEC-108 | Command Line Execution through SQL Injection
- cwe_id:CWE-108 | Struts: Unvalidated Action Form
- cwe_id:CWE-1389 | Incorrect Parsing of Numbers with Different Radices

Case 3 (global)
query: "CAPEC-101 server side include injection"
topk: 6
results:
- capec_id:CAPEC-101 | Server Side Include (SSI) Injection
- cwe_id:CWE-97 | Improper Neutralization of Server-Side Includes (SSI) Within a Web Page
- capec_id:CAPEC-14 | Client-side Injection-induced Buffer Overflow
- cwe_id:CWE-602 | Client-Side Enforcement of Server-Side Security
- cwe_id:CWE-603 | Use of Client-Side Authentication
- capec_id:CAPEC-664 | Server Side Request Forgery

## cwe_id

Case 1 (global)
query: "sql injection xss"
topk: 4
results:
- capec_id:CAPEC-7 | Blind SQL Injection
- capec_id:CAPEC-66 | SQL Injection
- capec_id:CAPEC-110 | SQL Injection through SOAP Parameter Tampering
- cwe_id:CWE-564 | SQL Injection: Hibernate

Case 2 (global)
query: "CWE-79 CWE-89 CWE-787"
topk: 5
results:
- cwe_id:CWE-787 | Out-of-bounds Write
- cwe_id:CWE-79 | Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')
- cwe_id:CWE-89 | Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')
- capec_id:CAPEC-89 | Pharming
- capec_id:CAPEC-79 | Using Slashes in Alternate Encoding

Case 3 (global)
query: "CWE-119 buffer overflow"
topk: 6
results:
- cwe_id:CWE-119 | Improper Restriction of Operations within the Bounds of a Memory Buffer
- cwe_id:CWE-122 | Heap-based Buffer Overflow
- cwe_id:CWE-680 | Integer Overflow to Buffer Overflow
- cwe_id:CWE-121 | Stack-based Buffer Overflow
- cwe_id:CWE-120 | Buffer Copy without Checking Size of Input ('Classic Buffer Overflow')
- cwe_id:CWE-131 | Incorrect Calculation of Buffer Size

## mitigation_id

Case 1 (global)
query: "code signing"
topk: 4
results:
- attack_technique_id:T1553.002 | Code Signing
- capec_id:CAPEC-206 | Signing Malicious Code
- capec_id:CAPEC-68 | Subvert Code-signing Facilities
- attack_technique_id:T1587.002 | Code Signing Certificates

Case 2 (global)
query: "M1031 M1037 M1053"
topk: 5
results:
- mitigation_id:M1053 | Data Backup
- mitigation_id:M1037 | Filter Network Traffic
- mitigation_id:M1031 | Network Intrusion Prevention
- attack_technique_id:T1071 | Application Layer Protocol
- attack_technique_id:T1071.004 | DNS

Case 3 (global)
query: "M1052 user account control"
topk: 6
results:
- mitigation_id:M1052 | User Account Control
- attack_technique_id:T1548.002 | Bypass User Account Control
- attack_technique_id:T1550.002 | Pass the Hash
- attack_technique_id:T1574.005 | Executable Installer File Permissions Weakness
- attack_technique_id:T1574.010 | Services File Permissions Weakness
- attack_technique_id:T1546.011 | Application Shimming

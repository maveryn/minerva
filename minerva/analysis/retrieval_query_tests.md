Retrieval query sanity checks (local engine)

Notes:
- Runs against the local RetrievalEngine (not the HTTP server).
- Exact ID extraction is enabled for all label types.
- Results are capped to the requested topk.

## attack_tactic_id

Case 1 (keywords-only)
query: "privilege escalation tactic"
topk: 4
results:
- attack_tactic_id:TA0004 | Privilege Escalation
- attack_tactic_id:TA0006 | Credential Access
- attack_tactic_id:TA0002 | Execution
- attack_tactic_id:TA0040 | Impact

Case 2 (ids-only)
query: "TA0004 TA0006 TA0008"
topk: 5
results:
- attack_tactic_id:TA0004 | Privilege Escalation
- attack_tactic_id:TA0006 | Credential Access
- attack_tactic_id:TA0008 | Lateral Movement
- attack_tactic_id:TA0002 | Execution
- attack_tactic_id:TA0040 | Impact

Case 3 (mixed)
query: "TA0003 persistence"
topk: 6
results:
- attack_tactic_id:TA0003 | Persistence
- attack_tactic_id:TA0006 | Credential Access
- attack_tactic_id:TA0002 | Execution
- attack_tactic_id:TA0040 | Impact
- attack_tactic_id:TA0004 | Privilege Escalation
- attack_tactic_id:TA0008 | Lateral Movement

## attack_technique_id

Case 1 (keywords-only)
query: "spearphishing attachment malicious file"
topk: 4
results:
- attack_technique_id:T1566.001 | Spearphishing Attachment
- attack_technique_id:T1598.002 | Spearphishing Attachment
- attack_technique_id:T1204.002 | Malicious File
- attack_technique_id:T1566.002 | Spearphishing Link

Case 2 (ids-only)
query: "T1566 T1566.001 T1566.002"
topk: 5
results:
- attack_technique_id:T1566 | Phishing
- attack_technique_id:T1566.001 | Spearphishing Attachment
- attack_technique_id:T1566.002 | Spearphishing Link
- attack_technique_id:T1204.001 | Malicious Link
- attack_technique_id:T1204.002 | Malicious File

Case 3 (mixed)
query: "T1059.003 powershell scripting"
topk: 6
results:
- attack_technique_id:T1059.003 | Windows Command Shell
- attack_technique_id:T1059 | Command and Scripting Interpreter
- attack_technique_id:T1059.001 | PowerShell
- attack_technique_id:T1546.013 | PowerShell Profile
- attack_technique_id:T1059.009 | Cloud API
- attack_technique_id:T1059.007 | JavaScript

## capec_id

Case 1 (keywords-only)
query: "buffer overflow environment variables"
topk: 4
results:
- capec_id:CAPEC-10 | Buffer Overflow via Environment Variables
- capec_id:CAPEC-46 | Overflow Variables and Tags
- capec_id:CAPEC-100 | Overflow Buffers
- capec_id:CAPEC-433 | Target Influence via The Human Buffer Overflow

Case 2 (ids-only)
query: "CAPEC-10 CAPEC-100 CAPEC-108"
topk: 5
results:
- capec_id:CAPEC-10 | Buffer Overflow via Environment Variables
- capec_id:CAPEC-100 | Overflow Buffers
- capec_id:CAPEC-108 | Command Line Execution through SQL Injection
- capec_id:CAPEC-622 | Electromagnetic Side-Channel Attack
- capec_id:CAPEC-438 | Modification During Manufacture

Case 3 (mixed)
query: "CAPEC-101 server side include injection"
topk: 6
results:
- capec_id:CAPEC-101 | Server Side Include (SSI) Injection
- capec_id:CAPEC-14 | Client-side Injection-induced Buffer Overflow
- capec_id:CAPEC-664 | Server Side Request Forgery
- capec_id:CAPEC-39 | Manipulating Opaque Client-based Data Tokens
- capec_id:CAPEC-110 | SQL Injection through SOAP Parameter Tampering
- capec_id:CAPEC-183 | IMAP/SMTP Command Injection

## cwe_id

Case 1 (keywords-only)
query: "sql injection xss"
topk: 4
results:
- cwe_id:CWE-564 | SQL Injection: Hibernate
- cwe_id:CWE-89 | Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')
- cwe_id:CWE-692 | Incomplete Denylist to Cross-Site Scripting
- cwe_id:CWE-566 | Authorization Bypass Through User-Controlled SQL Primary Key

Case 2 (ids-only)
query: "CWE-79 CWE-89 CWE-787"
topk: 5
results:
- cwe_id:CWE-79 | Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')
- cwe_id:CWE-89 | Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')
- cwe_id:CWE-787 | Out-of-bounds Write
- cwe_id:CWE-391 | Unchecked Error Condition
- cwe_id:CWE-373 | DEPRECATED: State Synchronization Error

Case 3 (mixed)
query: "CWE-119 buffer overflow"
topk: 6
results:
- cwe_id:CWE-119 | Improper Restriction of Operations within the Bounds of a Memory Buffer
- cwe_id:CWE-122 | Heap-based Buffer Overflow
- cwe_id:CWE-680 | Integer Overflow to Buffer Overflow
- cwe_id:CWE-121 | Stack-based Buffer Overflow
- cwe_id:CWE-120 | Buffer Copy without Checking Size of Input ('Classic Buffer Overflow')
- cwe_id:CWE-131 | Incorrect Calculation of Buffer Size

## detection_id

Case 1 (keywords-only)
query: "powershell execution detection"
topk: 4
results:
- detection_id:DET0455 | Abuse of PowerShell for Arbitrary Execution
- detection_id:DET0440 | Detecting PowerShell Execution via SyncAppvPublishingServer.vbs Proxy Abuse
- detection_id:DET0451 | Detection Strategy for PowerShell Profile Persistence via profile.ps1 Modification
- detection_id:DET0384 | Behavioral Detection of Unix Shell Execution

Case 2 (ids-only)
query: "DET0455 DET0830 DET0826"
topk: 5
results:
- detection_id:DET0455 | Abuse of PowerShell for Arbitrary Execution
- detection_id:DET0830 | Detection of Active Scanning
- detection_id:DET0826 | Detection of Gather Victim Host Information
- detection_id:DET0237 | Detection Strategy for Boot or Logon Initialization Scripts: RC Scripts
- detection_id:DET0535 | Detect Abuse of vSphere Installation Bundles (VIBs) for Persistent Access

Case 3 (mixed)
query: "DET0237 boot or logon initialization scripts"
topk: 6
results:
- detection_id:DET0237 | Detection Strategy for Boot or Logon Initialization Scripts: RC Scripts
- detection_id:DET0112 | Boot or Logon Initialization Scripts Detection Strategy
- detection_id:DET0274 | Boot or Logon Autostart Execution Detection Strategy
- detection_id:DET0150 | Detection Strategy for File Creation or Modification of Boot Files
- detection_id:DET0072 | Detect Logon Script Modifications and Execution
- detection_id:DET0582 | Detection Strategy for T1542.005 Pre-OS Boot: TFTP Boot

## mitigation_id

Case 1 (keywords-only)
query: "code signing"
topk: 4
results:
- mitigation_id:M1045 | Code Signing
- mitigation_id:M1038 | Execution Prevention
- mitigation_id:M1048 | Application Isolation and Sandboxing
- mitigation_id:M1025 | Privileged Process Integrity

Case 2 (ids-only)
query: "M1031 M1037 M1053"
topk: 5
results:
- mitigation_id:M1031 | Network Intrusion Prevention
- mitigation_id:M1037 | Filter Network Traffic
- mitigation_id:M1053 | Data Backup
- mitigation_id:M1016 | Vulnerability Scanning
- mitigation_id:M1035 | Limit Access to Resource Over Network

Case 3 (mixed)
query: "M1052 user account control"
topk: 6
results:
- mitigation_id:M1052 | User Account Control
- mitigation_id:M1018 | User Account Management
- mitigation_id:M1015 | Active Directory Configuration
- mitigation_id:M1026 | Privileged Account Management
- mitigation_id:M1036 | Account Use Policies
- mitigation_id:M1017 | User Training

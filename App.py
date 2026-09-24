import os
import re
import mimetypes
import streamlit as st

# Set up page configurations
st.set_page_config(
    page_title="RepoGuard - Static Malware Scanner",
    page_icon="🛡️",
    layout="wide"
)

# Base dictionary of high-risk static code indicators
SUSPICIOUS_PATTERNS = {
    "Obfuscated / Encoded Payloads": [
        r"base64\.b64decode", r"b64decode\(", r"eval\s*\(", r"exec\s*\("
    ],
    "Command Execution & Shells": [
        r"subprocess\.(Popen|run|call)", r"os\.system\(", r"pty\.spawn", r"/bin/sh", r"/bin/bash"
    ],
    "Network / Exfiltration Triggers": [
        r"requests\.(post|get)", r"urllib\.request", r"socket\.socket", r"curl\s+", r"wget\s+"
    ],
    "Suspicious Persistence / Hooks": [
        r"reg add", r"crontab", r"autorun", r"schtasks"
    ]
}

# Regex compilation for faster execution
COMPILED_PATTERNS = {category: [re.compile(p, re.IGNORECASE) for p in patterns] 
                     for category, patterns in SUSPICIOUS_PATTERNS.items()}

# High risk extensions often bundled in fake or malicious source repos
RISK_EXTENSIONS = {'.exe', '.dll', '.bat', '.sh', '.vbs', '.scr', '.bin', '.elf', '.msi'}

def scan_file(file_path):
    """Analyze a single file for known malicious extensions and text indicators."""
    findings = []
    _, ext = os.path.splitext(file_path.lower())
    
    # 1. Check Extensions
    if ext in RISK_EXTENSIONS:
        findings.append({
            "type": "Critical File Threat",
            "detail": f"High-risk compilation/executable file detected: `{ext}`",
            "line": "N/A"
        })
    
    # 2. Check Text Contents (Static Signatures)
    try:
        # Avoid reading massive binary logs or packages entirely to prevent crashes
        if os.path.getsize(file_path) > 10 * 1024 * 1024:  # Skip files > 10MB
            return findings
            
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                clean_line = line.strip()
                for category, regex_list in COMPILED_PATTERNS.items():
                    for regex in regex_list:
                        if regex.search(clean_line):
                            findings.append({
                                "type": category,
                                "detail": f"Matched pattern: `{regex.pattern}`",
                                "line": f"Line {line_num}: {clean_line[:100]}"
                            })
    except Exception as e:
        # Fail gracefully if file reading encounters issues
        pass
        
    return findings

def scan_directory(target_path):
    """Walk through the cloned repository tree and log security findings."""
    report = []
    total_files = 0
    
    for root, _, files in os.walk(target_path):
        # Exclude internal git configuration logs
        if '.git' in root:
            continue
        for file in files:
            total_files += 1
            full_path = os.path.join(root, file)
            file_findings = scan_file(full_path)
            if file_findings:
                report.append({
                    "file": os.path.relpath(full_path, target_path),
                    "issues": file_findings
                })
                
    return report, total_files

# --- App UI Layout ---
st.title("🛡️ RepoGuard: Cloned Repository Code Check")
st.markdown(
    "Verify the safety of local or newly cloned Git repositories before running or installing their code."
)

# Target input box
target_dir = st.text_input(
    "Enter absolute path to the local cloned folder:", 
    placeholder="e.g., /Users/username/projects/cloned-repo"
)

if st.button("Start Analysis Scan", type="primary"):
    if not target_dir:
        st.warning("Please enter a valid directory path path.")
    elif not os.path.isdir(target_dir):
        st.error("The path specified does not exist or is not a valid folder directory.")
    else:
        with st.spinner("Analyzing code architecture and scanning codebase files..."):
            scan_results, total_scanned = scan_directory(target_dir)
            
        st.success(f"Scan complete! Analyzed **{total_scanned}** total files.")
        
        if not scan_results:
            st.balloons()
            st.success("🎉 No high-risk static payloads, hidden executables, or obfuscation triggers found!")
        else:
            st.error(f"⚠️ Found potential indicators or anomalies in **{len(scan_results)}** files.")
            
            # Display findings grouped by file path
            for match in scan_results:
                with st.expander(f"📁 File: {match['file']} ({len(match['issues'])} alerts)"):
                    for issue in match['issues']:
                        st.markdown(f"**Threat Category:** `{issue['type']}`")
                        st.markdown(f"**Detail:** {issue['detail']}")
                        st.code(issue['line'], language='python')
                        st.write("---")


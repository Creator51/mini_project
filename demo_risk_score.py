"""
Demo script to test and display the Mathematical Risk Score Model evaluation directly in the terminal.
Formula: RS = min(100, alpha*C1 + beta*C2 + gamma*C3 + delta*C4)
Output Risk Level: INFO -> LOW -> MEDIUM -> HIGH -> CRITICAL
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.realtime.inference import RealTimeThreatDetector


def run_demo():
    print("=" * 68)
    print("      CYBER THREAT DETECTION FRAMEWORK - MATHEMATICAL RISK MODEL DEMO  ")
    print("=" * 68)
    print("Initializing ML Threat Detection Engine...\n")

    detector = RealTimeThreatDetector()

    test_samples = [
        {
            "name": "Sample 1: SQL Injection Payload Attack",
            "instruction": "SELECT * FROM users WHERE id=1 OR '1'='1",
            "port": "8080",
            "method": "POST",
            "ack": 250.0
        },
        {
            "name": "Sample 2: Ransomware SMB Encryption Vector",
            "instruction": "SMB Payload Drop: Encrypting volume with .locked extension",
            "port": "445",
            "method": "0.0",
            "ack": 80.0
        },
        {
            "name": "Sample 3: Normal Benign HTTP Web Traffic",
            "instruction": "GET /index.html HTTP/1.1 sensor reading clean",
            "port": "80",
            "method": "GET",
            "ack": 10.0
        }
    ]

    for sample in test_samples:
        print(f"\n>>> Running Evaluation for: {sample['name']}")
        detector.analyze_custom_instruction(
            text=sample["instruction"],
            port=sample["port"],
            method=sample["method"],
            ack=sample["ack"]
        )

if __name__ == "__main__":
    run_demo()

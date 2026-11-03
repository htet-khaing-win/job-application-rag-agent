"""CLI for the Job Application Assistant.

Usage:
    python main.py                 run the assistant
    python main.py add-key         store an API key in the encrypted vault (prompted, hidden)
    python main.py list-keys       show stored key aliases (masked)
    python main.py remove-key NAME delete a stored key
GUI:  streamlit run app.py
"""
import asyncio
import getpass
import sys

from pipeline import run_pipeline
from providers import PROVIDERS
from runtime import Profile, Runtime
from security import install_log_redaction
from vault import KeyVault, VaultError


def _runtime() -> Runtime:
    return Runtime(KeyVault(), Profile.load())


def cmd_add_key() -> None:
    vault = KeyVault()
    provider = input(f"Provider {list(PROVIDERS)}: ").strip().lower()
    if provider not in PROVIDERS or not PROVIDERS[provider].needs_key:
        print("Unknown provider or provider needs no key.")
        return
    alias = input("Alias (e.g. work-openai): ").strip() or provider
    secret = getpass.getpass("API key (hidden): ")
    persist = input("Save encrypted to disk? [Y/n]: ").strip().lower() != "n"
    try:
        vault.set(alias, provider, secret, persist=persist)
        print(f"Stored '{alias}'." + ("" if persist else " (session only; lost on exit)"))
    except VaultError as e:
        print(f"Error: {e}")


def cmd_list_keys() -> None:
    for k in KeyVault().list():
        print(f"{k['alias']:<24}{k['provider']:<12}{k['masked']}")


async def run() -> None:
    print("--- Job Application Assistant ---")
    print("Paste your Job Description below. Type 'DONE' on a new line when finished.")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "DONE":
            break
        lines.append(line)
    jd = "\n".join(lines)
    company = input("Please enter the company name: ").strip()

    try:
        result = await run_pipeline(_runtime(), jd, company)
    except ValueError as e:
        print(f"Cannot start: {e}\nAdd keys with `python main.py add-key` or use the GUI (streamlit run app.py).")
        return
    except Exception as e:  # noqa: BLE001
        from security import redact_secrets
        print(f"Error during processing: {redact_secrets(e)}")
        return

    if result.get("healing_log"):
        print("\nSelf-healing events:")
        for ev in result["healing_log"]:
            print("  -", ev)
    if result.get("is_fallback"):
        print("FALLBACK RESPONSE:\n" + result.get("final_response", "An error occurred."))
        return
    print("\nYour Cover Letter is Ready\n---------------------------------\n")
    print(result["cover_letter"])
    print(f"\n---------------------------------\nRefinement iterations: {result['refinement_count']}")


if __name__ == "__main__":
    install_log_redaction()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "add-key":
        cmd_add_key()
    elif cmd == "list-keys":
        cmd_list_keys()
    elif cmd == "remove-key" and len(sys.argv) > 2:
        print("Removed." if KeyVault().delete(sys.argv[2]) else "No such key.")
    else:
        asyncio.run(run())

"""RAG test: file embed hone ke baad sawal test.
Chalane ka tareeqa:  python test_rag.py
"""
import sys
from rag import ask, list_files

QUESTIONS = [
    "What is the confidentiality policy for employees?",
    "Notice period kitna hai?",
    "Annual leave ke baare mein kya likha hai?",
    "Company ki spaceship policy kya hai?",  # file mein nahi - 'nahi mili' aana chahiye
]

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else None
    print("Embed files:", list_files())
    for q in QUESTIONS:
        r = ask(q, source=source)
        print(f"\nQ: {q}\nA: {r['answer']}\nSources: {r['sources']}")
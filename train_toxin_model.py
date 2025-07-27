from Bio import SeqIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- Step 1: Load FASTA sequences ---
def read_fasta(file_path):
    return [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]

toxic_seqs = read_fasta("toxin.fasta")
non_toxic_seqs = read_fasta("non_toxin.fasta")

X = toxic_seqs + non_toxic_seqs
y = [1]*len(toxic_seqs) + [0]*len(non_toxic_seqs)

# --- Step 2: Vectorizer ---
def kmer_analyzer(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

# Use named function instead of lambda
vectorizer = TfidfVectorizer(analyzer=kmer_analyzer)
X_vec = vectorizer.fit_transform(X)

# --- Step 3: Train model ---
model = RandomForestClassifier(random_state=42)
model.fit(X_vec, y)

# --- Step 4: Save model and vectorizer ---
joblib.dump(model, "toxicity_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved.")

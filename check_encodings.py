import pickle

# Load the encodings file
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

# Display summary
print("✅ Encodings file loaded successfully!")
print(f"Total faces encoded: {len(data['encodings'])}")
print(f"Unique names found: {len(set(data['names']))}")
print("\nList of all registered students:\n")

# Show all unique names
for i, name in enumerate(sorted(set(data["names"])), start=1):
    print(f"{i}. {name}")

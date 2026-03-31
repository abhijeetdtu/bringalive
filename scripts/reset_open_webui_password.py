import sqlite3
import bcrypt
from getpass import getpass

#C:\Users\abhij\miniforge3\envs\llmui\Lib\site-packages\open_webui\data
# Change this to your Open WebUI DB path if needed
DB_PATH = r"webui.db"

email = input("Admin email: ").strip()
new_password = getpass("New password: ")

# Generate bcrypt hash
hashed = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt(rounds=10)).decode("utf-8")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("UPDATE auth SET password = ? WHERE email = ?", (hashed, email))
conn.commit()

if cur.rowcount == 0:
    print("No user found with that email.")
else:
    print("Password updated successfully.")

conn.close()
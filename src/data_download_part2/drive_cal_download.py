import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── Configuration ─────────────────────────────────────────────────────────
OUT_DIR  = r"data\raw\covariables\california"
SCOPES   = ['https://www.googleapis.com/auth/drive.readonly']

# Chemin vers credentials.json (racine du projet)
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = BASE_DIR   # ← même dossier que drive_cal_download.py
CRED_FILE = os.path.join(ROOT_DIR, 'credentials.json')
TOK_FILE  = os.path.join(ROOT_DIR, 'token.json')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Authentification ──────────────────────────────────────────────────────
def get_credentials():
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    creds = None

    # Token déjà sauvegardé
    if os.path.exists(TOK_FILE):
        creds = Credentials.from_authorized_user_file(TOK_FILE, SCOPES)

    # Token expiré ou absent → re-authentifier
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CRED_FILE):
                raise FileNotFoundError(
                    f"❌ credentials.json introuvable : {CRED_FILE}\n"
                    "   → Télécharge-le depuis console.cloud.google.com"
                )
            flow  = InstalledAppFlow.from_client_secrets_file(CRED_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # Sauvegarder le token
        with open(TOK_FILE, 'w') as f:
            f.write(creds.to_json())
        print(f"✅ Token sauvegardé : {TOK_FILE}")

    return creds

# ── Main ──────────────────────────────────────────────────────────────────
print("=" * 55)
print("Drive Download — CAL_CLIM_T* CSV")
print("=" * 55)

credentials = get_credentials()
service     = build('drive', 'v3', credentials=credentials)

# Chercher tous les CSV CAL_CLIM_T*
results = service.files().list(
    q="name contains 'CAL_CLIM_T' and trashed=false",
    fields="files(id, name)",
    pageSize=200
).execute()

files = results.get('files', [])
print(f"\n{len(files)} fichiers trouvés sur Drive")

if len(files) == 0:
    print("❌ Aucun fichier trouvé")
    print("   → Vérifie que les exports GEE sont terminés")
    print("   → Vérifie le nom : doit contenir 'CAL_CLIM_T'")
    exit()

# Trier par nom
files = sorted(files, key=lambda x: x['name'])

# ── Téléchargement ────────────────────────────────────────────────────────
downloaded = 0
skipped    = 0
failed     = 0

for i, f in enumerate(files):
    out_path = os.path.join(OUT_DIR, f['name'])

    if os.path.exists(out_path):
        print(f"⏭  [{i+1:02d}/{len(files)}] Déjà présent : {f['name']}")
        skipped += 1
        continue

    try:
        request    = service.files().get_media(fileId=f['id'])
        with open(out_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        print(f"✅ [{i+1:02d}/{len(files)}] {f['name']}")
        downloaded += 1

    except Exception as e:
        print(f"❌ [{i+1:02d}/{len(files)}] Échec {f['name']} : {e}")
        failed += 1

# ── Résumé ────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"✅ Téléchargés : {downloaded}")
print(f"⏭  Déjà présents : {skipped}")
print(f"❌ Échecs : {failed}")
print(f"📁 Dossier : {os.path.abspath(OUT_DIR)}")
print(f"{'='*55}")
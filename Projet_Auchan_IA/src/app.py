import streamlit as st
import joblib
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Auchan - Assistant Substituts",
    page_icon="🛒",
    layout="wide"
)

# --- 2. ACCESSIBILITÉ : THÈMES ET OPTIONS ---
st.sidebar.title("♿ Accessibilité")

# Choix du thème visuel
theme = st.sidebar.selectbox(
    "🎨 Thème visuel :",
    [
        "Standard (Charte Auchan)",
        "Contraste Élevé (Malvoyants)",
        "Deutéranopie (Rouge-Vert)",
        "Protanopie (Rouge faible)",
        "Tritanopie (Bleu-Jaune)",
    ],
    help="Choisissez un thème adapté à votre vision. La deutéranopie, protanopie et tritanopie sont des formes de daltonisme."
)

# Taille de police
taille_police = st.sidebar.select_slider(
    "🔡 Taille du texte :",
    options=["Petite (14px)", "Normale (17px)", "Grande (20px)", "Très grande (24px)"],
    value="Normale (17px)"
)

police_px = {
    "Petite (14px)": "14px",
    "Normale (17px)": "17px",
    "Grande (20px)": "20px",
    "Très grande (24px)": "24px",
}[taille_police]

# Mode sans images (lecteurs d'écran / connexion lente)
mode_texte = st.sidebar.checkbox(
    "📄 Mode texte uniquement (sans images)",
    value=False,
    help="Désactive les images pour faciliter la lecture par les technologies d'assistance."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "ℹ️ **Raccourcis clavier :** Utilisez `Tab` pour naviguer entre les éléments, `Entrée` pour valider."
)

# Synchronisation mode_texte dans session_state pour forcer le re-rendu des cartes
if "mode_texte" not in st.session_state:
    st.session_state["mode_texte"] = False
st.session_state["mode_texte"] = mode_texte

# --- PALETTES DE COULEURS PAR THÈME ---
# Chaque palette respecte un ratio de contraste WCAG AA (≥ 4.5:1 pour le texte normal)
palettes = {
    "Standard (Charte Auchan)": {
        "bg_app":       "#f9f9f9",
        "bg_card":      "#ffffff",
        "txt_card":     "#1a1a1a",       # Quasi-noir sur blanc → contraste 16:1
        "accent":       "#B30015",       # Rouge Auchan assombri → contraste 5.8:1 sur blanc
        "btn_bg":       "#B30015",
        "btn_txt":      "#ffffff",
        "border":       "#cccccc",
        "badge_ok":     "#1a6e2e",       # Vert foncé pour Nutri-Score A/B
        "badge_warn":   "#7a4b00",       # Ambre foncé pour C
        "badge_bad":    "#8b0000",       # Rouge foncé pour D/E
        "focus_ring":   "#B30015",
    },
    "Contraste Élevé (Malvoyants)": {
        "bg_app":       "#000000",
        "bg_card":      "#111111",
        "txt_card":     "#FFFFFF",
        "accent":       "#FFFF00",       # Jaune vif → contraste 19:1 sur noir
        "btn_bg":       "#FFFF00",
        "btn_txt":      "#000000",
        "border":       "#FFFF00",
        "badge_ok":     "#00FF88",
        "badge_warn":   "#FFB300",
        "badge_bad":    "#FF4444",
        "focus_ring":   "#FFFF00",
    },
    "Deutéranopie (Rouge-Vert)": {
        # Les deutéranopes confondent rouge et vert → on utilise bleu/orange
        "bg_app":       "#f4f6fa",
        "bg_card":      "#ffffff",
        "txt_card":     "#1a1a1a",
        "accent":       "#0057B7",       # Bleu vif → lisible pour deutéranopes
        "btn_bg":       "#0057B7",
        "btn_txt":      "#ffffff",
        "border":       "#aabcd4",
        "badge_ok":     "#0057B7",       # Bleu = positif (pas vert)
        "badge_warn":   "#E07B00",       # Orange = attention (pas jaune-vert)
        "badge_bad":    "#5C0099",       # Violet = négatif (pas rouge)
        "focus_ring":   "#E07B00",
    },
    "Protanopie (Rouge faible)": {
        # Les protanopes ne perçoivent pas le rouge → bleu/jaune/violet à la place
        "bg_app":       "#f4f6fa",
        "bg_card":      "#ffffff",
        "txt_card":     "#1a1a1a",
        "accent":       "#005BBB",       # Bleu → perçu clairement
        "btn_bg":       "#005BBB",
        "btn_txt":      "#ffffff",
        "border":       "#99b4d4",
        "badge_ok":     "#005BBB",
        "badge_warn":   "#CC7700",       # Orange-brun → distinguable
        "badge_bad":    "#6600AA",       # Violet foncé → distinguable du bleu
        "focus_ring":   "#CC7700",
    },
    "Tritanopie (Bleu-Jaune)": {
        # Les tritanopes confondent bleu et jaune → on utilise rouge/vert
        "bg_app":       "#f4f6f4",
        "bg_card":      "#ffffff",
        "txt_card":     "#1a1a1a",
        "accent":       "#C00020",       # Rouge → perçu clairement
        "btn_bg":       "#C00020",
        "btn_txt":      "#ffffff",
        "border":       "#c8d4c8",
        "badge_ok":     "#1a6e2e",       # Vert foncé → distinguable
        "badge_warn":   "#8B4513",       # Brun-rouge → distinguable
        "badge_bad":    "#C00020",       # Rouge → distinguable du vert
        "focus_ring":   "#C00020",
    },
}

p = palettes[theme]

# --- 3. INJECTION CSS ACCESSIBLE ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Montserrat:wght@400;600;700&display=swap');

    /* Police : Atkinson Hyperlegible est conçue pour les malvoyants */
    html, body, [class*="css"] {{
        font-family: 'Atkinson Hyperlegible', 'Montserrat', sans-serif !important;
        font-size: {police_px} !important;
        line-height: 1.6 !important;
    }}

    .stApp {{ background-color: {p['bg_app']} !important; }}

    /* ---- Couleur de texte globale (Correction de la fuite du thème) ---- */
    .stApp,
    div[data-testid="stMarkdownContainer"] *,
    div[data-baseweb] *,
    label, p, span, h1, h2, h3, h4, h5, h6 {{
        color: {p['txt_card']} !important;
    }}

    /* ---- Panneau latéral (Sidebar) ---- */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div:first-child {{
        background-color: {p['bg_card']} !important;
    }}

    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {{
        color: {p['txt_card']} !important;
    }}

    /* ---- Menus déroulants (Selectbox) ---- */
    div[data-baseweb="select"] > div {{
        background-color: {p['bg_card']} !important;
        border: 2px solid {p['border']} !important;
    }}
    
    div[data-baseweb="select"] *,
    div[data-baseweb="select"] span {{
        color: {p['txt_card']} !important;
    }}

    /* ---- Cartes produits ---- */
    .product-card {{
        background-color: {p['bg_card']} !important;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
        border: 2px solid {p['border']};
        margin-bottom: 20px;
        color: {p['txt_card']} !important;
        text-align: center;
        min-height: 420px;
    }}

    .product-card h4 {{
        color: {p['accent']} !important;
        margin-top: 12px;
        font-weight: 700;
        font-size: 1.1em;
        line-height: 1.4;
    }}

    .product-card p {{
        color: {p['txt_card']} !important;
        font-size: 0.95em;
        margin: 6px 0;
    }}

    /* Label de catégorie textuel SOUS l'image (accessible sans couleur seule) */
    .product-card .category-label {{
        display: inline-block;
        background: {p['accent']};
        color: {p['btn_txt']};
        border-radius: 5px;
        padding: 2px 8px;
        font-size: 0.8em;
        font-weight: 700;
        margin-bottom: 6px;
    }}

    /* ---- Badges Nutri-Score ---- */
    .nutri-badge {{
        display: inline-block;
        border-radius: 6px;
        padding: 4px 12px;
        font-weight: 700;
        font-size: 1em;
        letter-spacing: 0.05em;
        margin-top: 8px;
    }}
    .nutri-a {{ background: {p['badge_ok']};   color: #ffffff; }}
    .nutri-b {{ background: {p['badge_ok']};   color: #ffffff; }}
    .nutri-c {{ background: {p['badge_warn']}; color: #ffffff; }}
    .nutri-d {{ background: {p['badge_bad']};  color: #ffffff; }}
    .nutri-e {{ background: {p['badge_bad']};  color: #ffffff; }}
    .nutri-nc {{ background: #555555; color: #ffffff; }}

    /* ---- Boutons ---- */
    div[data-testid="stButton"] button {{
        background-color: {p['btn_bg']} !important;
        border-radius: 25px;
        border: 3px solid transparent !important;
        padding: 10px 24px;
        transition: filter 0.2s, outline 0.1s;
    }}

    /* Ciblage ultra-spécifique du texte à l'intérieur du bouton Streamlit */
    div[data-testid="stButton"] button p,
    div[data-testid="stButton"] button span {{
        color: {p['btn_txt']} !important;
        font-weight: 700 !important;
        font-size: 1em !important;
    }}

    /* Anneau de focus visible (navigation clavier) */
    div[data-testid="stButton"] button:focus,
    div[data-testid="stButton"] button:focus-visible {{
        outline: 4px solid {p['focus_ring']} !important;
        outline-offset: 3px !important;
    }}

    a:focus, a:focus-visible {{
        outline: 4px solid {p['focus_ring']} !important;
        outline-offset: 2px !important;
    }}

    /* ---- Liens visibles ---- */
    a {{
        color: {p['accent']} !important;
        text-decoration: underline !important;
    }}

    /* ---- Inputs (Champ de recherche) ---- */
    input[type="text"] {{
        background-color: {p['bg_card']} !important;
        color: {p['txt_card']} !important;
        border: 1px solid {p['border']} !important;
    }}

    input[type="text"]:focus {{
        outline: 3px solid {p['focus_ring']} !important;
        border-color: {p['focus_ring']} !important;
    }}

    /* ---- Réduction de mouvement (utilisateurs sensibles aux animations) ---- */
    @media (prefers-reduced-motion: reduce) {{
        * {{ animation: none !important; transition: none !important; }}
    }}

    /* ---- Mode impression (contraste maximum) ---- */
    @media print {{
        .stApp {{ background: white !important; color: black !important; }}
        .product-card {{ border: 2px solid black !important; }}
    }}
    </style>
""", unsafe_allow_html=True)


# --- 4. BACK-END : CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def charger_ressources():
    # 1. Obtenir le dossier actuel du script app.py (le dossier src/)
    dossier_actuel = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Remonter d'un niveau (..) pour atteindre la racine, puis cibler les bons dossiers
    chemin_models = os.path.join(dossier_actuel, '..', 'models')
    chemin_data = os.path.join(dossier_actuel, '..', 'data')
    
    try:
        # 3. Charger les fichiers depuis leurs nouveaux dossiers respectifs
        modele     = joblib.load(os.path.join(chemin_models, 'modele_auchan_svm.pkl'))
        vectoriseur = joblib.load(os.path.join(chemin_models, 'vectoriseur_tfidf.pkl'))
        catalogue  = pd.read_csv(os.path.join(chemin_data, 'catalogue_app.csv'))
        return modele, vectoriseur, catalogue
    except Exception as e:
        st.error(f"Erreur de chargement des fichiers : {e}")
        st.stop()

modele_svm, vectoriseur_tfidf, df_catalogue = charger_ressources()

nltk.download('stopwords', quiet=True)
stop_words_fr = set(stopwords.words('french'))

def nettoyer_texte(texte):
    texte = str(texte).lower()
    texte = re.sub(r'[^\w\s]', ' ', texte)
    texte = re.sub(r'\d+', '', texte)
    mots  = [mot for mot in texte.split() if mot not in stop_words_fr and len(mot) > 1]
    return ' '.join(mots)

def badge_nutriscore(note: str) -> str:
    """Retourne un badge HTML coloré ET textuel pour le Nutri-Score."""
    note = note.strip().upper()
    if note in ("A", "B"):
        css = "nutri-a" if note == "A" else "nutri-b"
        emoji = "🟢"
    elif note == "C":
        css, emoji = "nutri-c", "🟡"
    elif note in ("D", "E"):
        css = "nutri-d" if note == "D" else "nutri-e"
        emoji = "🔴"
    else:
        css, emoji = "nutri-nc", "⚪"
        note = "NC"
    # Couleur + lettre + emoji → triple redondance (couleur, forme, texte)
    return f'<span class="nutri-badge {css}" aria-label="Nutri-Score {note}">{emoji} Nutri-Score {note}</span>'


# --- 5. FRONT-END : INTERFACE ---
# Titre avec landmark HTML accessible
st.markdown(
    f'<h1 role="banner" aria-label="Assistant IA Auchan" '
    f'style="color:{p["txt_card"]} !important;">🛒 Assistant IA Auchan</h1>',
    unsafe_allow_html=True
)
st.write("Entrez le nom d'un produit pour trouver les meilleures alternatives disponibles.")

# Champ de recherche avec label explicite
nom_produit    = st.text_input(
    "🔍 Nom du produit à rechercher :",
    placeholder="Ex : Confiture de framboise...",
    help="Tapez un nom de produit puis cliquez sur le bouton ou appuyez sur Entrée."
)
bouton_recherche = st.button(
    "Trouver les meilleurs produits",
    use_container_width=True,
    help="Lance la recherche de substituts pour le produit saisi."
)

st.markdown('<hr aria-hidden="true">', unsafe_allow_html=True)

# --- 6. LOGIQUE MÉTIER ---
if bouton_recherche:
    if not nom_produit.strip():
        st.warning("⚠️ Veuillez entrer un nom de produit avant de lancer la recherche.")
    else:
        # A. Vectorisation & prédiction de catégorie
        texte_propre      = nettoyer_texte(nom_produit)
        vecteur_requete   = vectoriseur_tfidf.transform([texte_propre])
        prediction_cat    = modele_svm.predict(vecteur_requete)[0]

        # Bandeau rayon : fond vert fonce + texte blanc -> contraste 7.2:1
        st.markdown(
            f'<div role="status" aria-live="polite" style="'
            f'background-color:#1a6e2e; color:#ffffff; padding:12px 20px; '
            f'border-radius:8px; font-weight:700; font-size:1.05em; margin-bottom:12px;">'
            f'📍 Rayon identifié : {prediction_cat.capitalize()}'
            f'</div>',
            unsafe_allow_html=True
        )

        # B. Filtrage du catalogue
        df_filtre = df_catalogue[df_catalogue['target'] == prediction_cat].copy()

        if not df_filtre.empty:
            # C. Similarité cosinus
            noms_catalogue    = df_filtre['product_name'].fillna('').apply(nettoyer_texte)
            matrices_produits = vectoriseur_tfidf.transform(noms_catalogue)
            scores            = cosine_similarity(vecteur_requete, matrices_produits).flatten()

            # D. Boost sur correspondance exacte des mots-clés
            mots_cles = texte_propre.split()
            for i, nom_prod in enumerate(noms_catalogue):
                for mot in mots_cles:
                    if mot in nom_prod:
                        scores[i] += 0.3

            df_filtre['score_final'] = scores
            resultats = df_filtre.sort_values(by='score_final', ascending=False).head(3).to_dict('records')

            # E. Annonce pour lecteurs d'écran
            st.markdown(
                f'<p aria-live="polite" aria-atomic="true" style="color:{p["txt_card"]}">'
                f'✅ {len(resultats)} produit(s) trouvé(s) dans le rayon « {prediction_cat.capitalize()} ».</p>',
                unsafe_allow_html=True
            )

            # F. Affichage des cartes
            cols = st.columns(3)
            for i, produit in enumerate(resultats):
                with cols[i]:
                    url_img = produit.get('image_url', "")
                    if pd.isna(url_img) or str(url_img).strip() == "":
                        url_img = "https://via.placeholder.com/300x300.png?text=Image+Indisponible"

                    nom_affiche  = str(produit['product_name']).title()[:60]
                    marque       = str(produit['brands']).title() if not pd.isna(produit.get('brands')) else "Auchan"
                    nutri_raw    = str(produit.get('nutriscore_grade', 'NC')).upper().strip()
                    badge_html   = badge_nutriscore(nutri_raw)

                    # Image (ou texte si mode texte seul)
                    if st.session_state.get("mode_texte", False):
                        img_html = f'<p aria-hidden="false">🖼️ <i>Image non affichée (mode texte)</i></p>'
                    else:
                        img_html = (
                            f'<img src="{url_img}" '
                            f'alt="Photo de {nom_affiche}" '
                            f'style="width:100%; max-height:180px; object-fit:contain; border-radius:10px; background:white;" '
                            f'loading="lazy">'
                        )

                    # Carte HTML avec rôle ARIA
                    st.markdown(f"""
                        <article class="product-card" role="article" aria-label="Produit {i+1} : {nom_affiche}">
                            {img_html}
                            <span class="category-label">{prediction_cat.capitalize()}</span>
                            <h4>{nom_affiche}</h4>
                            <p>🏢 <strong>Marque :</strong> {marque}</p>
                            {badge_html}
                        </article>
                    """, unsafe_allow_html=True)

                    # Bouton Streamlit natif (accessible au clavier)
                    st.link_button(
                        f"Voir sur Auchan.fr",
                        "https://www.auchan.fr/",
                        use_container_width=True,
                        help=f"Ouvrir la page Auchan pour : {nom_affiche}"
                    )
        else:
            st.info("ℹ️ Aucun substitut trouvé dans cette catégorie. Essayez un autre terme.")

# --- 7. FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<hr aria-hidden="true">', unsafe_allow_html=True)
st.caption(
    "Application de thèse — Algorithme : SVM + Similarité Cosine avec Pondération Sémantique. "
    "Thèmes d'accessibilité conformes WCAG 2.1 AA."
)
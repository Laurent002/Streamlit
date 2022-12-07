import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
import plotly.express as px
import numpy as np
from PIL import Image


@st.cache(suppress_st_warning=True)
def lit_données():
        chemin_accès = "s&p500.csv"
        data = pd.read_csv(chemin_accès)
        return data

def paramètres(df, secteur_par_défaut, capitalisation_par_défaut):

    ### Secteur ###
    secteurs = [secteur_par_défaut] + list(df['sector'].unique())
    secteur = st.sidebar.selectbox("Secteur d'activité", secteurs, index=0)

    ### Capitalisation ###
    liste_capitalisation = [capitalisation_par_défaut] + ["Petite", "Moyenne", "Grande"]
    capitalisation = st.sidebar.selectbox("Capitalisation", liste_capitalisation, index=0)

    ### Dividende ###
    dividende = st.sidebar.slider("Taux de dividende entre : (%)", 0.0, 10.0, value=(0.0, 10.0))

    ### Profit ###
    profit_min, profit_max = float(df['profitMargins_%'].min()), float(df['profitMargins_%'].max())
    profit = st.sidebar.slider("Marge de profit plus grande que : (%)", profit_min, profit_max, step=10.0)

    return secteur, capitalisation, dividende, profit


def filtrage(df, secteur_par_défaut, capitalisation_par_défaut, secteur, capitalisation, dividende, profit):
#    filtering(df_sp,sector_default_val,cap_default_val,option_sector,dividend_value,profit_value,cap_value):
 
     ### Secteur ###
    if secteur != secteur_par_défaut:
        df = df[(df['sector'] == secteur)]
        

    ### Capitalisation ###
    if capitalisation != capitalisation_par_défaut:
        if capitalisation == 'Petite':
            df = df[(df['marketCap'] >= 0) & (df['marketCap'] <= 20e9)]
        elif capitalisation == 'Moyenne':
            df = df[(df['marketCap'] > 20e9) & (df['marketCap'] <= 100e9)]
        elif capitalisation == 'Grande':
            df = df[(df['marketCap'] > 100e9)]



    ### Dividende
    df = df[(df['dividendYield_%'] >= dividende[0]) & (df['dividendYield_%'] <= dividende[1])]

    ### Profit ###
    df = df[(df['profitMargins_%'] >= profit)]

    return df

def prix_entreprise(df, option_entreprise):
    if option_entreprise != None:
        ticker_entreprise = df.loc[df['name'] == option_entreprise, 'ticker'].values[0]
        donnée_prix = pdr.get_data_yahoo(ticker_entreprise, start='2011-12-31', end = '2022-12-01')['Adj Close']
        donnée_prix = donnée_prix.reset_index(drop=False)
        donnée_prix.columns = ['ds', 'y']
        return donnée_prix
    return None
    

def évolution_court_action(prix):
    figure = px.line(prix, x='ds', y='y', title="Court de l'action sur 10 ans")
    figure.update_xaxes(title_text = 'Date')
    figure.update_yaxes(title_text = "Prix de l'action")
    st.plotly_chart(figure)
    return

def  métriques(prix):
    prix_action_2012 = prix.loc[prix['ds'] == '2012-01-03', 'y'].values[0]
    prix_action_2022 = prix.loc[prix['ds'] == '2022-12-01', 'y'].values[0]
    performance = np.around((prix_action_2022/prix_action_2012 - 1)*100, 2)
    return prix_action_2022, performance



if __name__ == "__main__":
    st.set_page_config(
        page_title="Mon projet Streamlit",
        page_icon="📈",
        initial_sidebar_state="expanded"
    )

st.title("Analyse et visualisation S&P500")
st.sidebar.title("Critères de recherche")

image = Image.open("stock.jpeg")
_, colonne_image_2, _ = st.columns([1,3,1])
with colonne_image_2:
       st.image(image, caption='@austindistel')

df = lit_données()

secteur_par_défaut = 'ALL'
capitalisation_par_défaut = 'ALL'
secteur, capitalisation, dividende, profit = paramètres(df, secteur_par_défaut, capitalisation_par_défaut)
df = filtrage(df, secteur_par_défaut, capitalisation_par_défaut, secteur, capitalisation, dividende, profit)

st.subheader("Partie 1 - Visualisation des données")
with st.expander("Partie 1 - Explication"):
    st.write("""
            Dans la table ci-dessous, vous trouverez la plupart des entreprises du S&P500 (stock market index of the 500 largest American companies) avec certains critères tels que :
                
                - Le nom de la compagnie
                - Le secteur d'activité
                - La capitalisation du marché
                - Le pourcentage de dividende (dividend/stock price)
                - La marge de profit de la compagnie en pourcentage
            
            ⚠️ Ces données sont prélevées de l'API Yahoo finance. ⚠️

            ℹ️ Vous pouvez chercher et filter par compagnie avec le filtre sur la gauche. ℹ️
          """)

st.write("")
st.write("Nombre d'entreprises dans la table: ", len(df)) 
st.dataframe(df.iloc[:,1:])

### Partie 2 - Choisir une entreprise et afficher ses informations ne temps réel ###

st.subheader("Partie 2 - Choisissez une entreprise")
option_entreprise = st.selectbox("Entreprises:", df.name.unique())


### Partie 3 - Afficher les infos en temps réel sur l'entreprise sélectionnée précédemment

st.subheader("Partie 3 - Analyse du court de l'action pour l'entreprise {}".format(option_entreprise))
prix = prix_entreprise(df, option_entreprise)
# st.dataframe(prix)


### Graphique de l'évolution du prix de l'action ###

évolution_court_action(prix)
prix_action_2022, performance = métriques(prix)
col_prédiction_1, col_prédiction_2 = st.columns([1,2])

with col_prédiction_1:
    st.metric(label = "Prix de l'action fin 2022", value=str(np.around(prix_action_2022,2)), delta=str(performance) + '%')
    st.write("Comparaison avec le 3 janvier 2012")

with col_prédiction_2:
    with st.expander("Explication de l'analyse", expanded=True):
        st.write ("""
                    Le graphe ci-dessus montre l'évolution de l'action pour l'entreprise choisie 
                    entre le 31 décembre 2011 et le 1er décembre 2022.

                    L'indicateur est le prix de l'action au 1er décembre 2022 ainsi que son évolution durant la période spécifiée.

                    Ces données proviennent de l'API Yahoo Finance.
        """)

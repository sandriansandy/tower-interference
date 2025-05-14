import streamlit as st
import pandas as pd
import joblib, sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan scaler
model = joblib.load('./model_xgb.pkl')
scaler = joblib.load('./scaler_model.pkl')
feature_names = joblib.load('./feature_names.pkl')

def preprocess_input(input_df):
    input_df = input_df.drop(columns=['Nomor_Tower', 'Ruas_Pengantar'], errors='ignore')
    categorical_cols = input_df.select_dtypes(include='object').columns
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    missing_cols = set(feature_names) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[feature_names]
    scaled_data = scaler.transform(input_df)
    return scaled_data

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Gangguan Tower",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dictionary teks untuk multi-bahasa
texts = {
    "id": {
        "title": "🏗️ Prediksi Gangguan Tower Listrik",
        "desc": "Aplikasi ini memprediksi kemungkinan gangguan pada tower listrik berdasarkan parameter teknis",
        "menu_header": "📋 Menu Input Data",
        "menu_method": "Pilih metode input data:",
        "method_options": ('Single Prediction', 'Batch Prediction'),
        "parameter_title": "🔧 Parameter Teknis",
        "single_result_title": "📊 Hasil Prediksi",
        "prob_label": "Probabilitas Gangguan",
        "prob_help": "Persentase kemungkinan terjadinya gangguan",
        "pred_ada": "🚨 Prediksi: ADA GANGGUAN",
        "pred_tidak": "✅ Prediksi: TIDAK ADA GANGGUAN",
        "feature_title": "📌 Faktor yang Mempengaruhi",
        "batch_note": "Data yang dimasukkan kolomnya harus sesuai dan urut, seperti ini",
        "upload_prompt": "Upload file CSV/XLSX",
        "upload_success": "File berhasil diupload!",
        "preview_data": "Preview Data:",
        "predict_button": "🚀 Lakukan Prediksi",
        "download_result": "📥 Download Hasil Prediksi",
        "info_title": "ℹ️ Informasi Aplikasi",
        "info_content": """**Aplikasi Prediksi Gangguan Tower** ini menggunakan model machine learning (XGBoost) 
        yang telah dilatih untuk memprediksi kemungkinan gangguan pada tower listrik 
        berdasarkan parameter teknis seperti:
        - Usia aset
        - Riwayat gangguan sebelumnya
        - Parameter arus listrik
        - Nilai pentanahan
        - Dan parameter teknis lainnya
        
        **Cara penggunaan:**
        1. Pilih metode input (single/batch)
        2. Isi/upload data
        3. Klik tombol prediksi
        4. Lihat hasil dan analisis"""
    },
    "en": {
        "title": "🏗️ Power Tower Failure Prediction",
        "desc": "This app predicts the probability of failure on power towers based on technical parameters",
        "menu_header": "📋 Input Data Menu",
        "menu_method": "Choose input method:",
        "method_options": ('Single Prediction', 'Batch Prediction'),
        "parameter_title": "🔧 Technical Parameters",
        "single_result_title": "📊 Prediction Result",
        "prob_label": "Failure Probability",
        "prob_help": "Percentage chance of failure occurring",
        "pred_ada": "🚨 Prediction: FAILURE DETECTED",
        "pred_tidak": "✅ Prediction: NO FAILURE",
        "feature_title": "📌 Influencing Factors",
        "batch_note": "Your data columns should match and be in order like this",
        "upload_prompt": "Upload CSV/XLSX file",
        "upload_success": "File uploaded successfully!",
        "preview_data": "Data Preview:",
        "predict_button": "🚀 Predict",
        "download_result": "📥 Download Prediction Result",
        "info_title": "ℹ️ About This App",
        "info_content": """**This Power Tower Failure Prediction App** uses a machine learning model (XGBoost) 
        trained to predict the probability of failures on power towers based on technical parameters like:
        - Asset age
        - Past failure history
        - Electric current parameters
        - Grounding values
        - And other technical parameters
        
        **How to use:**
        1. Choose input method (single/batch)
        2. Fill/upload data
        3. Click predict button
        4. See results and analysis"""
    }
}

# Pilih bahasa
with st.sidebar:
    lang = st.selectbox("🌐 Pilih Bahasa / Select Language", options=["id", "en"])

# Judul & Deskripsi
st.title(texts[lang]["title"])
st.markdown(texts[lang]["desc"])

# Sidebar menu
with st.sidebar:
    st.header(texts[lang]["menu_header"])
    st.markdown(texts[lang]["menu_method"])
    input_method = st.radio("", texts[lang]["method_options"])

if input_method == texts[lang]["method_options"][0]:
    with st.expander(texts[lang]["parameter_title"], expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            count = st.number_input('Count', 0, 100, 0)
            count_minus = st.number_input('Count -', 0, 100, 0)
            count_plus = st.number_input('Count +', 0, 100, 0)
            percent_plus = st.number_input('% +', 0.0, 100.0, 0.0)
            density = st.number_input('Density', 0.0, 10.0, 1.0)
        with col2:
            min_ka = st.number_input('Min kA', -5000.0, 0.0)
            max_ka = st.number_input('Max kA', -5000.0, 0.0)
            mean_ka = st.number_input('Mean kA', -5000.0, 0.0)
            exposure_factor = st.number_input('Exposure Factor', -5000.0, 0.0)
            min_ka_minus = st.number_input('Min kA -', -5000.0, 0.0)
        with col3:
            max_ka_minus = st.number_input('Max kA -', -5000.0, 0.0)
            min_ka_plus = st.number_input('Min kA +', -5000.0, 0.0)
            max_ka_plus = st.number_input('Max kA +', -5000.0, 0.0)
            mean_ka_plus = st.number_input('Mean kA +', 0.0, 1000.0, 50.0)
            area = st.number_input('Area', 0.0, 1000.0, 500.0)

    input_data = pd.DataFrame([[count, count_minus, count_plus, percent_plus, density,
                                min_ka, max_ka, mean_ka, exposure_factor,
                                min_ka_minus, max_ka_minus, min_ka_plus, max_ka_plus,
                                mean_ka_plus, area]],
                              columns=['Count','Count_-','Count_+','%_+','Density',
                                       'Min_kA','Max_kA','Mean_kA','Exposure_factor',
                                       'Min_kA_-','Max_kA_-','Min_kA_+','Max_kA_+',
                                       'Mean_kA_+','Area'])

else:
    st.text(texts[lang]["batch_note"])
    st.text('Kategori_Nilai_Pentanahan | Usia_Aset | Count | Count_- | Count_+ | %_+ | Density | Min_kA | Max_kA | Mean_kA | Exposure_factor | Min_kA_- | Max_kA_- | Mean_kA_- | Min_kA_+ | Max_kA_+ | Mean_kA_+ | Area')
    uploaded_file = st.file_uploader(texts[lang]["upload_prompt"], type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            input_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success(texts[lang]["upload_success"])
            st.write(texts[lang]["preview_data"])
            st.dataframe(input_data.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

if st.button(texts[lang]["predict_button"]):
    try:
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)
        proba = model.predict_proba(processed_data)
        st.subheader(texts[lang]["single_result_title"])

        if input_method == texts[lang]["method_options"][0]:
            col1, col2 = st.columns(2)
            with col1:
                prob_gangguan = proba[0][1]*100
                st.metric(label=texts[lang]["prob_label"], 
                          value=f"{prob_gangguan:.1f}%",
                          help=texts[lang]["prob_help"])
                st.progress(int(prob_gangguan))
            with col2:
                st.error(texts[lang]["pred_ada"]) if prediction[0]==1 else st.success(texts[lang]["pred_tidak"])

            st.subheader(texts[lang]["feature_title"])
            fig, ax = plt.subplots()
            feature_importance = model.feature_importances_
            features = input_data.columns
            sns.barplot(x=feature_importance, y=features, palette="viridis", ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)

        else:
            input_data['Prediksi'] = prediction
            input_data['Probabilitas_Gangguan'] = proba[:,1]
            st.dataframe(input_data.style.background_gradient(subset=['Probabilitas_Gangguan'], cmap='Reds'))
            st.download_button(texts[lang]["download_result"],
                               data=input_data.to_csv(index=False).encode('utf-8'),
                               file_name='hasil_prediksi.csv',
                               mime='text/csv')

    except Exception as e:
        st.error(f"Terjadi error dalam pemrosesan: {e}")

with st.expander(texts[lang]["info_title"]):
    st.markdown(texts[lang]["info_content"])

st.markdown("---")
st.caption("© 2025 Tim Prediksi Gangguan Tower BDA Kelompok 6. All rights reserved.")

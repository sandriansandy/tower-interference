import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan scaler
model = joblib.load('model_xgb.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

def preprocess_input(input_df):
    # Lakukan preprocessing sama seperti di training
    # 1. Mapping manual untuk kolom tertentu
    input_df['Kategori_Nilai_Pentanahan'] = input_df['Kategori_Nilai_Pentanahan'].str.lower().map({'buruk':0, 'baik':1})
    
    # 2. Drop kolom yang sama
    input_df = input_df.drop(columns=['Nomor_Tower', 'Ruas_Pengantar'], errors='ignore')
    
    # 3. Encoding categorical variables
    categorical_cols = input_df.select_dtypes(include='object').columns
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    
    # 4. Align columns dengan data training
    missing_cols = set(feature_names) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Tambahkan kolom yang hilang dengan nilai 0
    
    # 5. Urutkan kolom sesuai urutan training
    input_df = input_df[feature_names]
    
    # 6. Scaling
    scaled_data = scaler.transform(input_df)
    return scaled_data

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Gangguan Tower",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul aplikasi
st.title('🏗️ Prediksi Gangguan Tower Listrik')
st.markdown("Aplikasi ini memprediksi kemungkinan gangguan pada tower listrik berdasarkan parameter teknis")

with st.sidebar:
    st.header("📋 Menu Input Data")
    st.markdown("Pilih metode input data:")
    input_method = st.radio(
        "Metode Input:",
        ('Single Prediction', 'Batch Prediction')
    )

if input_method == 'Single Prediction':
    with st.expander("🔧 Parameter Teknis", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            kategori_pentanahan = st.selectbox('Kategori Pentanahan', ['Baik', 'Buruk'])
            usia_aset = st.slider('Usia Aset (tahun)', 0, 100, 5)
            count = st.number_input('Count', 0, 100, 0)
            count_minus = st.number_input('Count -', 0, 100, 0)
            count_plus = st.number_input('Count +', 0, 100, 0)
            percent_plus = st.number_input('% +', 0.0, 100.0, 0.0)
            
        with col2:
            density = st.number_input('Density', 0.0, 10.0, 1.0)
            min_ka = st.number_input('Min kA', min_value=-5000.0, value=0.0)
            max_ka = st.number_input('Max kA', min_value=-5000.0, value=0.0)
            mean_ka = st.number_input('Mean kA', min_value=-5000.0, value=0.0)
            exposure_factor = st.number_input('Exposure Factor', min_value=-5000.0, value=0.0)
            min_ka_minus = st.number_input('Min kA -', min_value=-5000.0, value=0.0)
            
        with col3:
            max_ka_minus = st.number_input('Max kA -', min_value=-5000.0, value=0.0)
            mean_ka_minus = st.number_input('Mean kA -', min_value=-5000.0, value=0.0)
            min_ka_plus = st.number_input('Min kA +', min_value=-5000.0, value=0.0)
            max_ka_plus = st.number_input('Max kA +', min_value=-5000.0, value=0.0)
            mean_ka_plus = st.number_input('Mean kA +', 0.0, 1000.0, 50.0)
            area = st.number_input('Area', 0.0, 1000.0, 500.0)

    # Membuat dataframe input dengan URUTAN SESUAI TRAINING
    input_data = pd.DataFrame([[
        kategori_pentanahan,      # Kategori_Nilai_Pentanahan
        usia_aset,                # Usia_Aset
        count,                    # Count
        count_minus,              # Count_-
        count_plus,               # Count_+
        percent_plus,             # %_+
        density,                  # Density
        min_ka,                   # Min_kA
        max_ka,                   # Max_kA
        mean_ka,                  # Mean_kA
        exposure_factor,          # Exposure_factor
        min_ka_minus,             # Min_kA_-
        max_ka_minus,             # Max_kA_-
        mean_ka_minus,            # Mean_kA_-
        min_ka_plus,              # Min_kA_+
        max_ka_plus,              # Max_kA_+
        mean_ka_plus,             # Mean_kA_+
        area                      # Area
    ]], columns=[
        'Kategori_Nilai_Pentanahan',
        'Usia_Aset',
        'Count',
        'Count_-',
        'Count_+',
        '%_+',
        'Density',
        'Min_kA',
        'Max_kA',
        'Mean_kA',
        'Exposure_factor',
        'Min_kA_-',
        'Max_kA_-',
        'Mean_kA_-',
        'Min_kA_+',
        'Max_kA_+',
        'Mean_kA_+',
        'Area'
    ])

else:
    st.text("Data yang dimasukkan kolomnya harus sesuai dan urut, seperti ini")
    st.text('Kategori_Nilai_Pentanahan | Usia_Aset | Count | Count_- | Count_+ | %_+ | Density | Min_kA | Max_kA | Mean_kA | Exposure_factor | Min_kA_- | Max_kA_- | Mean_kA_- | Min_kA_+ | Max_kA_+ | Mean_kA_+ | Area')
    uploaded_file = st.file_uploader("Upload file CSV/XLSX", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                input_data = pd.read_csv(uploaded_file)
            else:
                input_data = pd.read_excel(uploaded_file)
            st.success("File berhasil diupload!")
            st.write("Preview Data:")
            st.dataframe(input_data.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

if st.button('🚀 Lakukan Prediksi'):
    try:
        # Preprocessing
        processed_data = preprocess_input(input_data)
        
        # Predict
        prediction = model.predict(processed_data)
        proba = model.predict_proba(processed_data)
        
        # Visualisasi hasil
        st.subheader("📊 Hasil Prediksi")
        
        if input_method == 'Single Prediction':
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                # Progress bar untuk probabilitas
                prob_gangguan = proba[0][1]*100
                st.metric(label="Probabilitas Gangguan", 
                         value=f"{prob_gangguan:.1f}%",
                         help="Persentase kemungkinan terjadinya gangguan")
                st.progress(int(prob_gangguan))
                
            with col_result2:
                # Tampilkan hasil dengan warna
                if prediction[0] == 1:
                    st.error('🚨 Prediksi: ADA GANGGUAN')
                else:
                    st.success('✅ Prediksi: TIDAK ADA GANGGUAN')
                
            # Visualisasi feature importance
            st.subheader("📌 Faktor yang Mempengaruhi")
            fig, ax = plt.subplots()
            feature_importance = model.feature_importances_
            features = input_data.columns
            sns.barplot(x=feature_importance, y=features, palette="viridis", ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)
            
        else:
            # Untuk batch prediction
            input_data['Prediksi'] = prediction
            input_data['Probabilitas_Gangguan'] = proba[:,1]
            st.dataframe(input_data.style.background_gradient(
                subset=['Probabilitas_Gangguan'], 
                cmap='Reds'))
            
            # Download hasil
            st.download_button(
                label="📥 Download Hasil Prediksi",
                data=input_data.to_csv(index=False).encode('utf-8'),
                file_name='hasil_prediksi.csv',
                mime='text/csv')
            
    except Exception as e:
        st.error(f"Terjadi error dalam pemrosesan: {e}")

# Tambahan informasi
with st.expander("ℹ️ Informasi Aplikasi"):
    st.markdown("""
    **Aplikasi Prediksi Gangguan Tower** ini menggunakan model machine learning (XGBoost) 
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
    4. Lihat hasil dan analisis
    """)

st.markdown("---")
st.caption("© 2023 Tim Prediksi Gangguan Tower. All rights reserved.")
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image


def run():
    # Membuat Title
    st.title('Credit Card Payment Default Prediction')

    # Membuat Subtitle
    st.subheader('Exploratory Data Analysis')

    # Menampilkan gambar
    image = Image.open('bank.jpg')
    st.image(image, caption='Credit Card Payment Default Prediction')

    # Show DataFrame
    st.write('### Data Credit Card Payment Default')
    df = pd.read_csv('credit_card.csv')
    st.dataframe(df)

    # Distribution of Default Payment by Gender (Bar Chart)
    st.subheader('Distribution of Default Payments by Gender')
    default_by_gender = df.groupby('sex')['default_payment_next_month'].value_counts().unstack()
    fig, ax = plt.subplots(figsize=(8, 6))
    default_by_gender.plot(kind='bar', ax=ax)

    ax.set_title('Distribution of Default Payments by Gender')
    ax.set_xlabel('Gender (1: Male, 2: Female)')
    ax.set_ylabel('Number of Customers')
    ax.set_xticks(range(len(default_by_gender.index)))
    ax.set_xticklabels(['Male', 'Female'], rotation=0)
    ax.legend(title='Default Payment', labels=['No', 'Yes'])

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3, fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    st.write('''
    - Nasabah dengan jenis kelamin perempuan lebih banyak dibandingkan dengan laki-laki.
    - Nasabah perempuan juga lebih banyak mengalami gagal bayar dibandingkan laki-laki secara kuantitas.''')



    
    # Distribution of Default Payment by Gender (Pie Chart)
    st.write('#### Distribution of Default Payment by Gender')
    default_by_gender = df.groupby('sex')['default_payment_next_month'].value_counts().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    male_defaults = default_by_gender.loc[1]
    axes[0].pie(male_defaults, labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Default Payment Distribution for Males')
    axes[0].axis('equal')

    female_defaults = default_by_gender.loc[2]
    axes[1].pie(female_defaults, labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Default Payment Distribution for Females')
    axes[1].axis('equal')

    plt.tight_layout()
    st.pyplot(fig)

    st.write('''
    - Jika dilihat dari perbadingan tiap gender, nasabah laki-laki lebih besar mengalami gagal bayar dibanding nasabah perempuan walaupun dengan selisih yang tipis (±2%).''')
    st.markdown('---')



    # Distribution of Default Payment by Education Level (Bar Chart)
    st.subheader('Distribution of Default Payment by Education Level')
    default_by_education = df.groupby('education_level')['default_payment_next_month'].value_counts().unstack()
    default_by_education = default_by_education.loc[[1, 2, 3, 4]]

    fig, ax = plt.subplots(figsize=(10, 6))
    default_by_education.plot(kind='bar', ax=ax)
    ax.set_title('Distribution of Default Payment by Education Level')
    ax.set_xlabel('Education Level (1: Graduate School, 2: University, 3: High School, 4: Others)')
    ax.set_ylabel('Number of Customers')
    ax.set_xticks(range(len(default_by_education.index)))
    ax.set_xticklabels(['Graduate School', 'University', 'High School', 'Others'], rotation=0)
    ax.legend(title='Default Payment', labels=['No', 'Yes'])

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3, fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    st.write('''
    - Jika dilihat dari perbadingan tiap gender, nasabah laki-laki lebih besar mengalami gagal bayar dibanding nasabah perempuan walaupun dengan selisih yang tipis (±2%).''')
    st.markdown('---')




    # Distribution of Default Payment by Marital Status (Bar Chart)
    st.subheader('Distribution of Default Payments by Marital Status')
    marital_statuses = [1, 2, 3]
    default_by_marital_status = df[df['marital_status'].isin(marital_statuses)] \
        .groupby('marital_status')['default_payment_next_month'] \
        .value_counts().unstack()

    fig, ax = plt.subplots(figsize=(8, 6))
    default_by_marital_status.plot(kind='bar', ax=ax)

    ax.set_title('Distribution of Default Payments by Marital Status')
    ax.set_xlabel('Marital Status (1: Married, 2: Single, 3: Others)')
    ax.set_ylabel('Number of Customers')

    ax.set_xticks(range(len(marital_statuses)))
    ax.set_xticklabels(marital_statuses, rotation=0)

    ax.legend(title='Default Payment', labels=['No', 'Yes'])

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3, fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    #Distribution of Default Payment by Marital Status (Pie Chart)
    default_by_marital = df.groupby('marital_status')['default_payment_next_month'].value_counts().unstack()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    labels = {1: 'Married', 2: 'Single', 3: 'Others'}
    marital_statuses = [1, 2, 3]

    for i, status in enumerate(marital_statuses):
        if status in default_by_marital.index:
            marital_defaults = default_by_marital.loc[status]
            axes[i].pie(marital_defaults,
                        labels=['No Default', 'Default'],
                        autopct='%1.1f%%',
                        startangle=90)
            axes[i].set_title(f'Default Payment Distribution - {labels[status]}')
            axes[i].axis('equal')
        else:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[i].set_title(f'Default Payment Distribution - {labels[status]}')
            axes[i].axis('off')

    plt.tight_layout()
    fig.text(0.5, -0.03, 'Marital Status: 1 = Married, 2 = Single, 3 = Others',
            ha='center', fontsize=12)

    st.pyplot(fig)

    st.write('''
    - Nasabah yang belum menikah ('Single') jumlahnya paling banyak, namun persentasenya bukan yang paling besar.
    - Nasabah yang sudah menikah ('Married') merupakan nasabah yang secara persentase lebih besar dibandingkan yang belum menikah dan others.''')
    st.markdown('---')



    st.subheader('Distribution of Default Payments by Age')

    default_by_age = df.groupby('age')['default_payment_next_month'].value_counts().unstack()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get counts safely with .get() - assuming 0 = No Default, 1 = Default
    no_default = default_by_age.get(0, 0)
    default = default_by_age.get(1, 0)

    # Plot bars: No Default at bottom, Default stacked on top
    ax.bar(default_by_age.index, no_default, label='No Default')
    ax.bar(default_by_age.index, default, bottom=no_default, label='Default')

    ax.set_title('Distribution of Default Payments by Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of Customers')

    # Set xticks and labels properly
    ax.set_xticks(default_by_age.index)
    ax.set_xticklabels(default_by_age.index, rotation=45)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

    st.write('''
    - Hampir semua kelompok usia tidak lepas dari gagal bayar, menunjukkan bahwa risiko gagal bayar ada di seluruh rentang usia.
    - Jika dilihat dari usia, nasabah pengguna kartu kredit kebanyakan berada di rentang usia 23-42 tahun, dengan usia 30 yang paling banyak penggunanya.''')
    st.markdown('---')

    # Membuat Title
    st.write('Page ini dibuat oleh **Dendi Apriyandi**')

if __name__ == '__main__':
    run()
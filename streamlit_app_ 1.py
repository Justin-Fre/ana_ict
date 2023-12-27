import time  # to simulate a real time data, time loop
import random
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import pyodbc
import streamlit as st
from sklearn.preprocessing import LabelEncoder
coder = LabelEncoder()
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
lgbmodel = LGBMClassifier(random_state=42, num_class = 3)
import joblib
from Feature_engineering import *
import base64

###load gif###

file_ = open("ann.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()


booking_time = 20231031
########################################################
############# Cấu hình ETL dữ liệu #####################
########################################################
conn_str = (
    r'Driver={SQL Server};'
    r'Server=ps1biccapp04;'
    r'Trusted_Connection=yes;'
    )

cnxn = pyodbc.connect(conn_str)
#########################################################

model_trained = joblib.load(r"D:/Recomander_system/Final_app/saved_models/v6_5_10.pkl")

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

def highlight_pd_holding(s):
    return ['background-color: #008446']*len(s) if s.Flag else ['background-color: #EDEDED']*len(s)


# dashboard title
title_container = st.container()
col11, col22 = st.columns([0.2,0.7], gap='large')
 
with title_container:
    with col11:
        st.image('logo-VPBank.png', width=250)
    with col22:
        st.header('Competent Product Boosting (CPB) – CSAT & Booking Online')
        st.caption('***For a prosperous Vietnam***')

st.markdown('''
    :blue[ Thông tin chung: Demo Chương trình gợi ý bán sản phẩm tại Quầy đối với nhóm khách hàng tham gia đặt lịch trên hệ thống Booking Online của VPBank]''')

# top-level filters
col1, col2 = st.columns(2)

with col1:
    selected_customer = st.text_area("🗝️ Nhập CIF khách hàng tại đây:")


with col2:
    EDA = st.button('🟢 Danh sách mô hình gợi ý bán:')
    if EDA :
        st.markdown('''Vui lòng đợi trong quá trình mô hình khởi động ...''')
        st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,)
        #Caching the model for faster loading
        # @st.cache
        # if __name__ == '__main__':
        # Load data infer
        file_name_columns = 'D:/Recomander_system/Final_app/dataset/columns_name_trained.csv'
        train_columns = pd.read_csv(file_name_columns)

        file_name_infer = 'D:/Recomander_system/Final_app/dataset/T11_2023_input_infer.csv'
        data_infer = pd.read_csv(file_name_infer)

        # load model
        file_name_model_trained = r"D:/Recomander_system/Final_app/saved_models/v6_5_10.pkl"
        model_trained = joblib.load(r"D:/Recomander_system/Final_app/saved_models/v6_5_10.pkl")

        data_infer = data_infer.fillna(0)
        data_infer = cat_feature_processing(data_infer)

        # Xóa những cột không sử dụng
        unused_col = ['SERVICE_SUB_TYPE', 'BRANCH_CODE']
        data_infer = drop_unused_col(data_infer, unused_col)

        # Onehot transform với các cột dạng category
        cat_col = ('SEGMENT_FINAL', 'Nghe_nghiep', 'SERVICE_TYPE_FL', 'VPB_GENDER_FL', 'VPB_EDUCATION_FL')
        data_infer_onehot = onehot_transform(data_infer, cat_col)

        # Chuẩn hóa tập infer phù hợp với đầu vào của model_trained
        data_infer_input = standardize_input(data_infer_onehot, train_columns)
        df_output = predict_product(model_trained, data_infer, data_infer_input)
        # st.dataframe(df_output)

        ###############################
        df_prod = pd.read_csv("D:/Recomander_system/Final_app/dataset/data_product_holding.csv")

        df_prod['CIF'] = df_prod['CIF'].astype('str')
        df_prod = df_prod[df_prod['CIF'] == selected_customer]

        df_output['CIF'] = df_output['CIF'].astype('str')

        # Merge the dataframes on CIF and YEAR_MONTH
        merged_df = pd.merge(df_output, df_prod, on=['CIF'], how = 'left')

        columns_to_update = merged_df['PRODUCT'].dropna().unique()
        for column in columns_to_update:
            merged_df.loc[merged_df['PRODUCT'] == column, column] = 0

        merged_df.drop('PRODUCT', axis=1, inplace=True)
        merged_df = merged_df.groupby(['CIF']).min().reset_index()

        # Get the column name with the maximum value in each row
        merged_df['first_rec'] = merged_df.iloc[:, 1:8].idxmax(axis=1)

        merged_df['second_rec'] = merged_df.iloc[:, 1:8].apply(second_largest_column, axis=1)
        
        # merged_df
        merged_df_fn = merged_df[merged_df['CIF'] == selected_customer]
        st.write("Dựa trên hành vi của khách hàng, mô hình gợi ý bán sản phẩm cho khách hàng",selected_customer, "là:")
        st.write(" - 1.", merged_df_fn['first_rec'].values[0] if len(merged_df_fn['first_rec']) > 0 else None)
        st.write("Dựa trên hành vi của khách hàng, mô hình gợi ý bán sản phẩm cho khách hàng",selected_customer, "là:")
        st.write(" - 2.", merged_df_fn['second_rec'].values[0] if len(merged_df_fn['first_rec']) > 0 else None)

if st.button('🙍‍♂️ Nhấn để Khai thác chân dung khách hàng tại đây: 🔻'):

    # create three columns
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    # fill in those three columns with respective metrics or KPIs
    #
    query_demo = f'''
    select CIF, VPB_GENDER, SECTOR, EDUCATION, AGE, 
    CASE WHEN GOLD_CUS = 1 THEN 'AF' 
    WHEN SEGMENT = 'KHCN' AND MASS_AF = 'YES' THEN 'MAF'
	WHEN SEGMENT = 'KHCN' THEN 'MASS'
	ELSE 'OTHER' END AS SEGMENT_FINAL
	from CUSTOMER.dbo.VPB_CUSTOMER_202311
    WHERE CIF like (%s)
    ''' % selected_customer
    globals()[f'df_customer'] = pd.read_sql(query_demo, cnxn)

    #
    query_AUM = f'''
    select CIF, AUM_TYPE, TOTAL_AUM
	from CUSTOMER.dbo.AUM_CUSTOMER_202311
    WHERE CIF like (%s)
    ''' % selected_customer
    globals()[f'df_AUM'] = pd.read_sql(query_AUM, cnxn)

    #
    query_pd_holding = f'''
    select top 10 CIF, 
  				PROD_CA_PAYROLL as PAYROLL,
  				PROD_CA_NONE_PAYROLL AS CASA,
  				PROD_TD	AS FLAG_TD,
  				PROD_OTHER_SAVING AS SAVING,	
  				PROD_AUTO_LOAN AS VAY_MUA_XE,
  				PROD_HOME_LOAN AS VAY_MUA_NHA,
  				PROD_HOUSE_HOLD VAY_KINH_DOANH,
  				PROD_CONSUMP AS VAY_TIEU_DUNG,	
  				PROD_PASSBOOK AS VAY_STK,
  				PROD_SECURITIES AS VAY_CK,
  				PROD_UPL AS VAY_TIN_CHAP,
  				PROD_OVERDRAFT AS VAY_THAU_CHI,
  				PROD_OTH_LOAN AS VAY_KHAC,
  				PROD_CREDIT_CARD AS CREDIT_CARD,
  				PROD_INSURANCE AS FLAG_BAO_HIEM,
  				PROD_AIA AS FLAG_AIA,
  				PROD_DEBIT AS FLAG_DEBIT,
  				PROD_BOND AS FLAG_BOND,
  				PROD_MFUND AS FLAG_FUND,
  				PROD_SERVICE_TOTAL AS TOTAL_PD_HOLDING 
  			from CUSTOMER.dbo.CUSTOMER_PRODUCT_HOLDING_{booking_time}
    WHERE CIF like (%s)
    ''' % selected_customer
    globals()[f'df_pd_holding'] = pd.read_sql(query_pd_holding, cnxn)
    

    # st.table(df_customer['SEGMENT_FINAL'][0])
    kpi1.metric(
        label="Age ⏳",
        value=round(df_customer['AGE']),
    )
    
    kpi2.metric(
        label="Phân khúc 🎯",
        value= df_customer['SEGMENT_FINAL'][0],
    )
    
    kpi3.metric(
        label="Tổng tài sản (AUM - VND) 💰",
        value=f"💶 {round(df_AUM['TOTAL_AUM'][0]):,} ",
    )

    kpi4.metric(
    label="Tổng sản phẩm nắm giữ 🔎",
    value=df_pd_holding['TOTAL_PD_HOLDING'][0],
    )

    # create two columns for charts
    fig_col1, fig_col2 = st.columns(2)

    #
    query_TOI = f'''
    select *
	from MONTHLY_NEWDATA.dbo.RB_TOI_BY_CIF
    WHERE CAST(YEARMONTH AS INT) >= 202301 and CIF like (%s)
    ''' % selected_customer
    globals()[f'df_TOI'] = pd.read_sql(query_TOI, cnxn)

    #
    query_TXN = f'''
    select YEARMONTH, TXN_DEBIT_CREDIT, AMOUNT
    from MONTHLY_NEWDATA.dbo.TRANSACTION_CASA 
    WHERE CONVERT(INT, YEARMONTH) >= 202301 and CIF like (%s)
    ''' % selected_customer
    globals()[f'df_TXN'] = pd.read_sql(query_TXN, cnxn)

    df_TXN_FN = pd.pivot_table(df_TXN, values='AMOUNT', index=['YEARMONTH'],
                       columns=['TXN_DEBIT_CREDIT'], aggfunc="sum").reset_index()

    df_TXN_FN = df_TXN_FN.rename(columns = {'CR' : 'Dòng tiền chảy vào VPBank',
                                            'DB' : 'Dòng tiển chảy ra VPBank'})

    with fig_col1:
        st.markdown("### 💹 TOI của khách hàng:")
        fig = st.line_chart(df_TOI, x="YEARMONTH", 
        y=["TOI"], color=["#0000FF"])
        #st.write(fig)
        
    with fig_col2:
        st.markdown("### 📈 Xu hướng tiền vào/ tiền ra của khách hàng:")
        fig2 = st.line_chart(df_TXN_FN, x="YEARMONTH", y=["Dòng tiền chảy vào VPBank", "Dòng tiển chảy ra VPBank"], color=["#008446", "#ec2028"])
        # st.write(fig2)

    st.markdown("### ✔️ Chi tiết nắm giữ sản phẩm tại VPBank:")
    
    df_pd_holding_1 = df_pd_holding.drop(columns = ['CIF', 'TOTAL_PD_HOLDING'])
    st.table(df_pd_holding_1.transpose().reset_index().rename(columns={'index':'Sản phẩm', 0 : 'Flag'}).style.apply(highlight_pd_holding, axis=1))

    # EDA = st.checkbox('🟢 Danh sách mô hình gợi ý bán:')
    # if EDA :
    #     st.write("Dựa trên hành vi của khách hàng, mô hình gợi ý bán sản phẩm cho khách hàng",selected_customer, "là:")
    #     st.write(" - 1.", random.choice(["Term_Deposit", "Banca", "Qualified-Casa", "Bond", "Bank Neo", "Overdraft"]))
    #     st.write("Dựa trên hành vi của khách hàng, mô hình gợi ý bán sản phẩm cho khách hàng",selected_customer, "là:")
    #     st.write(" - 2.", random.choice(["Term_Deposit", "Banca", "Qualified-Casa", "Bond", "Bank Neo", "Overdraft"]))

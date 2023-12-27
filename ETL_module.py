import pyodbc
import pandas as pd

# Configure Connetion RBD
conn_str = (
    r'Driver={SQL Server};'
    r'Server=ps1biccapp04;'
    r'Trusted_Connection=yes;'
    )

cnxn = pyodbc.connect(conn_str)

# Configure Connetion SBD
conn_str_sbd = (
    r'Driver={SQL Server};'
    r'Server=ps1biccapp01;'
    r'Trusted_Connection=yes;'
    )

cnxn_sbd = pyodbc.connect(conn_str_sbd)

#### Set date_time_RBD
time = '20231130'

#### Set time for Booking online
booking_time = 202311

###################################
###### Dữ liệu Booking Online #####
###################################

query_booking = f'''
    select CONVERT(VARCHAR(6), TRY_CAST(CREATE_DATE_TIME AS datetime), 112) AS YEARMONTH, CUST_CODE as CIF,
		SERVICE_TYPE, SERVICE_SUB_TYPE, BRANCH_CODE
		from STAGING.DBO.CUST_BOOKING_ONL_INTF
			where CONVERT(VARCHAR(6), TRY_CAST(CREATE_DATE_TIME AS datetime), 112) = {booking_time}
    '''
globals()[f'df_booking_'] = pd.read_sql(query_booking, cnxn_sbd)

###################################
##### 1. Dữ liệu Demographic ######
###################################

query_demo = f'''
select CIF, VPB_GENDER, SECTOR, EDUCATION, AGE, 
        CASE WHEN GOLD_CUS = 1 THEN 'AF' 
      WHEN SEGMENT = 'KHCN' AND MASS_AF = 'YES' THEN 'MAF'
	  WHEN SEGMENT = 'KHCN' THEN 'MASS'
	  ELSE 'OTHER' END AS SEGMENT_FINAL
	  from CUSTOMER.dbo.VPB_CUSTOMER_202311
''' 
globals()[f'df_demo_'] = pd.read_sql(query_demo, cnxn)

###################################
##### 2. Dữ liệu Credit Card ######
###################################
query_cc = f'''
    select CONVERT(VARCHAR(6), TRY_CAST(BUSINESS_DATE AS datetime), 112) AS YEARMONTH, CIF, TOTAL_NUMBER_CC, TOTAL_BALANCE_CC,
		NEW_CC_P3M_INDICATOR, CLOSED_CC_P3M_INDICATOR, AVG_AMOUNT_CC_P3M, 
		AVG_GROWTH_AMOUNT_CC_13_VS_46, AVG_AMOUNT_CC_LOCAL_P3M, AVG_AMOUNT_CC_OVERSEAS_P3M, AVG_NUMBER_CC_P3M
		from ANALYTICS.dbo.CREDITCARD_{time}
    '''
globals()[f'df_CREDITCARD_{time}'] = pd.read_sql(query_cc, cnxn)

###################################
##### 3. Dữ liệu Deposit ##########
###################################

query_deposit = f'''
select CONVERT(VARCHAR(6), BUSINESS_DATE, 112) AS YEARMONTH, CIF, TOTAL_NUMBER_CASA
    AVG_GROWTH_AMOUNT_DEPOSIT,	AVG_BALANCE_CASA_P1M, AVG_BALANCE_CASA_P3M, TOTAL_NUMBER_TD, AVG_BALANCE_TD_P3M,
    TOTAL_NUMBER_TD_MATURE_NEXT_3M, AVG_BALANCE_FCY_P1M, 	TOTAL_NUMBER_FCY, 	AVG_BALANCE_FCY_P3M
    FROM ANALYTICS.dbo.DEPOSIT_{time}
'''
globals()[f'df_DEPOSIT_{time}'] = pd.read_sql(query_deposit, cnxn)

###################################
##### 4. Dữ liệu IDC ##############
###################################

query_idc = f'''
select CONVERT(VARCHAR(6), TRY_CAST(BUSINESS_DATE AS datetime), 112) AS YEARMONTH, CIF, TOTAL_NUMBER_IDC_P1M, TOTAL_AMOUNT_IDC_P1M, 
        AVG_NUMBER_IDC_P3M, AVG_GROWTH_AMOUNT_IDC_P3M
    FROM ANALYTICS.dbo.IDC_{time}
'''
globals()[f'df_IDC_{time}'] = pd.read_sql(query_idc, cnxn)

###################################
##### 5. Dữ liệu Insurance ########
###################################

query_ins = f'''
select CONVERT(VARCHAR(6), TRY_CAST(BUSINESS_DATE AS datetime), 112) AS YEARMONTH, BUSINESS_DATE, CIF, 	
    BANCA_EXCLUDE_CREDIT_LIFE_INDICATOR, 	BANCA_CREDIT_LIFE_INDICATOR, TOTAL_NUMBER_BANCA_LIFE, 	TOTAL_NUMBER_BANCA_HEALTH_CARE
    FROM ANALYTICS.dbo.INSURANCE_{time}
'''
globals()[f'df_Insurance_{time}'] = pd.read_sql(query_ins, cnxn)

###################################
##### 6. Dữ liệu Investment #######
###################################

query_inv = f'''
    select CONVERT(VARCHAR(6), TRY_CAST(BUSINESS_DATE AS datetime), 112) AS YEARMONTH, CIF, 	
        INVEST_INDICATOR, 	AVG_BALANCE_INVEST_P3M, AVG_GROWTH_BALANCE_INVEST, AVG_BALANCE_BOND_P3M, AVG_BALANCE_MUTUAL_FUND_P3M
    FROM ANALYTICS.dbo.INVESTMENT_{time}
'''
globals()[f'df_INVESTMENT_{time}'] = pd.read_sql(query_inv, cnxn)

###################################
##### 7. Dữ liệu Loan #############
###################################

query_loan = f'''
    select CONVERT(VARCHAR(6), TRY_CAST(BUSINESS_DATE AS datetime), 112) AS YEARMONTH, CIF,
    LOAN_INDICATOR, NEW_LOAN_P1M, NEW_LOAN_P3M, NEW_LOAN_P6M, 	LOAN_INSTALLMENT_INDICATOR, SECURED_LOAN_INDICATOR,
    PERSONAL_LOAN_INDICATOR, OVERDRAFT_INDICATOR, OTHER_LOAN_INDICATOR
    FROM ANALYTICS.dbo.LOAN_{time}
'''
globals()[f'df_LOAN_{time}'] = pd.read_sql(query_loan, cnxn)

###################################
##### 8. Dữ liệu Transaction ######
###################################

query_txn = f'''
    Select CONVERT(VARCHAR(6), TRY_CAST(BUSINESS_DATE AS datetime), 112) AS YEARMONTH, CIF,
    MOB, ONLINE_BANKING_CUS_INDICATOR, MOBILE_BANKING_CUS_INDICATOR, OTC_CUS_INDICATOR, NON_OTC_CUS_INDICATOR,
    AVG_NUMBER_OTC_P3M, AVG_GROWTH_NUMBER_OTC, AVG_NUMBER_CREDIT_OTC_P3M, AVG_NUMBER_DEBIT_OTC_P3M, AVG_AMOUNT_CREDIT_OTC_P3M,
    AVG_AMOUNT_DEBIT_OTC_P3M
    FROM ANALYTICS.dbo.TRANSACTION_{time}
'''
globals()[f'df_TRANSACTION_{time}'] = pd.read_sql(query_txn, cnxn)

df = df_booking_.merge(df_demo_, how='left', on=['CIF'])
df = df.merge(df_CREDITCARD_20231130.drop(columns = 'YEARMONTH'), how='left', on=['CIF'])
df = df.merge(df_DEPOSIT_20231130.drop(columns = 'YEARMONTH'), how='left', on=['CIF'])
df = df.merge(df_IDC_20231130.drop(columns = 'YEARMONTH'), how='left', on=['CIF'])
df = df.merge(df_Insurance_20231130.drop(columns = 'YEARMONTH'), how='left', on=['CIF'])
df = df.merge(df_INVESTMENT_20231130.drop(columns = 'YEARMONTH'), how='left', on=['CIF'])
df = df.merge(df_LOAN_20231130.drop(columns = 'YEARMONTH'), how='left', on=['CIF'])
df = df.merge(df_TRANSACTION_20231130.drop(columns = 'YEARMONTH'), how='left', on=['CIF'])
df.shape

#Def function JOB
def Nghe_nghiep(df):
    if df['SECTOR'] == '1001':
        return 'NVVP'
    elif df['SECTOR'] == '1900':
        return 'CN_Khac'
    elif df['SECTOR'] == '1007':
        return 'CCVC'
    elif df['SECTOR'] == '1008':
        return 'LD_PHOTHONG'
    elif df['SECTOR'] == '1010':
        return 'nghi_huu'
    elif df['SECTOR'] == '1009':
        return 'HSSV'
    elif df['SECTOR'] == '1004':
        return 'HGD'
    elif df['SECTOR'] == '1002':
        return 'VPB_ER'
    else:
        return 'other'
   
df['Nghe_nghiep'] = df.apply(Nghe_nghiep, axis = 1)
df = df.drop(columns = ['SECTOR'])

df = df.rename(columns = {'CIF' : 'CIF_FN'})

df.to_csv('D:/Recomander_system/Final_app/dataset/etl_final_data.csv', index = False)
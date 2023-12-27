import numpy as np 
import pandas as pd 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import joblib

# class feature_processing():
def cat_feature_processing(df):
    #Def function SERVICE_TYPE
    def SERVICE_TYPE_FL(df):
        if df['SERVICE_TYPE'] == 'Nộp, rút tiền mặt':
            return 'Nop_rut_tien'
        elif df['SERVICE_TYPE'] == 'Giao dịch không tiền mặt':
            return 'GD_KO_TIEN_MAT'
        elif (df['SERVICE_TYPE'] == 'Khác') or (df['SERVICE_TYPE'] == 0):
            return 'Khac'
        elif (df['SERVICE_TYPE'] == 'Tiết kiệm'):
            return 'Tiet_kiem'
        elif (df['SERVICE_TYPE'] == 'Chuyển khoản/ thanh toán/ trả nợ'):
            return 'Tra_No'
        elif (df['SERVICE_TYPE'] == 'CROSS SALE'):
            return 'cross_sale'
        elif (df['SERVICE_TYPE'] == 'Ký hồ sơ tín dụng'):
            return 'Ky_hso'

    #Def function GENDER
    def GENDER_FL(df):
        if df['VPB_GENDER'] == 'MALE':
            return 'MALE'
        elif df['VPB_GENDER'] == 'FEMALE':
            return 'FEMALE'
        else:
            return 'OTHER'
    
    df['SERVICE_TYPE_FL'] = df.apply(SERVICE_TYPE_FL, axis = 1)
    df['VPB_GENDER_FL'] = df.apply(GENDER_FL, axis = 1)
    
    #Def function EDUCATION
    df['VPB_EDUCATION_FL'] = np.where(df['EDUCATION']==0, 'other', df['EDUCATION'])
    df['VPB_EDUCATION_FL'] = df['VPB_EDUCATION_FL'].str.replace(r' ', '_')

    df = df.drop(['SERVICE_TYPE', 'VPB_GENDER', 'EDUCATION'], axis = 1)
    return df

def onehot_transform(df, cat_col):
    '''
        Onehot categories column
        Method: One-hot dummies
    '''
    for col in cat_col:
        if col in df.columns:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
            df = df.drop(col, axis=1)
    return df

def standardize_input(input_data, train_columns):
    '''
        Input: 2 tập dataframe X_input và columns_input
        Output: tập X_input có các cột cột khớp với tập dataframe cột đầu vào: columns_input

        So sánh các cột trong tập X_input với columns_input:
            * Sẽ gán các cột không có trong tập X_test bằng giá trị 0
            * Xóa các cột có trong tập X_test nhưng không có giá trị trong tập columns_input
    '''
    train_column_names = train_columns.iloc[:, 0].values.tolist()
    train_column_types = train_columns.iloc[:, 1].values.tolist()
    train_data_types = dict(zip(train_column_names, train_column_types))

    not_in_train = list(set(input_data.columns).difference(train_column_names))
    missing_train_column = list(set(train_column_names).difference(input_data.columns))
    input_data[missing_train_column] = 0
    input_data = input_data.drop(not_in_train, axis=1)
    input_data = input_data.reindex(columns=train_column_names)
    input_data = input_data.astype(train_data_types)
    return input_data

def drop_unused_col(df, unused_col):
    df = df.drop(unused_col, axis=1, errors='ignore')
    return df

def predict_product(model_trained, data_infer_cif, data_infer_input):
    y_predict_infer = model_trained.predict_proba(data_infer_input)
    product_name = ('BANK_NEO', 'Banca', 'OTHERS', 'QUALIFIED_CASA', 'THAU_CHI', 'TRAI_PHIEU', 'Term_Deposit')
    df_predict = pd.DataFrame(y_predict_infer, columns=product_name)
    df_output = pd.DataFrame({'CIF': data_infer_cif['CIF_FN'].astype(int).tolist()})
    df_output = pd.concat([df_output, df_predict], axis=1)
    return df_output


# Define a function to get the column name with the second-largest value
def second_largest_column(row):
    top_two = row.nlargest(2)
    return top_two.index[1]

# if __name__ == '__main__':
#     # Load data infer
#     file_name_columns = 'D:/Recomander_system/Final_app/dataset/columns_name_trained.csv'
#     train_columns = pd.read_csv(file_name_columns)

#     file_name_infer = 'D:/Recomander_system/Final_app/dataset/T11_2023_input_infer.csv'
#     data_infer = pd.read_csv(file_name_infer)

#     # load model
#     file_name_model_trained = "D:\Project\Product_rcm_sys\Implement\Model_trained\\v6_5_10.pkl"
#     model_trained = joblib.load(file_name_model_trained)

#     data_infer = data_infer.fillna(0)
#     data_infer = cat_feature_processing(data_infer)

#     # Xóa những cột không sử dụng
#     unused_col = ['SERVICE_SUB_TYPE', 'BRANCH_CODE']
#     data_infer = drop_unused_col(data_infer, unused_col)

#     # Onehot transform với các cột dạng category
#     cat_col = ('SEGMENT_FINAL', 'Nghe_nghiep', 'SERVICE_TYPE_FL', 'VPB_GENDER_FL', 'VPB_EDUCATION_FL')
#     data_infer_onehot = onehot_transform(data_infer, cat_col)

#     # Chuẩn hóa tập infer phù hợp với đầu vào của model_trained
#     # data_infer_input = standardize_input(data_infer_onehot, train_columns)

#     # df_output = predict_product(data_infer_input)

#     df_output = predict_product(data_infer_onehot)

#     df_output.to_csv('D:\Project\Product_rcm_sys\Implement\Model_trained\df_output.csv')
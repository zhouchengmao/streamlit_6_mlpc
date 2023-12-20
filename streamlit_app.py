import pandas as pd
import streamlit as st

from MLUtils import *

st.title("Dr. Z.C.M.")
st.title("Application of artificial intelligence technology to build a prediction model of postoperative depression in colon cancer")  # 算法名称 and XXX

COL_INPUT = None

vars = []

btn_predict = None

mlpc = None


# 配置选择变量（添加生成新数据并预测的功能）
def setup_selectors():
    global vars, btn_predict

    if COL_INPUT is not None and len(COL_INPUT) > 0:
        col_num = 3
        cols = st.columns(col_num)

        for i, c in enumerate(COL_INPUT):
            with cols[i % col_num]:
                num_input = st.number_input(f"Please input {c}", value=0, format="%d", key=c)
                vars.append(num_input)

        with cols[0]:
            btn_predict = st.button("Do Predict")

    if btn_predict:
        do_predict()


# 对上传的文件进行处理和展示
def do_processing():
    global COL_INPUT
    global mlpc
    pocd, col_input = read_csv('./pocd.csv')
    pocd_t, _ = read_csv('./pocd_t.csv')
    COL_INPUT = col_input

    st.text("Dataset Description")
    st.write(pocd.describe())
    if st.checkbox('Show detail of this dataset', key='cbDetail'):
        st.write(pocd)

    st.text("Dataset Description")
    st.write(pocd_t.describe())
    if st.checkbox('Show detail of this dataset', key="cbDetail2"):
        st.write(pocd_t)

    # 分割数据
    X_train, X_test, y_train, y_test = do_split_data_with_2files(pocd, pocd_t)
    X_train, X_test, y_train, y_test = do_xy_preprocessing(X_train, X_test, y_train, y_test)

    col1, col2 = st.columns(2)

    # 准备模型
    mlpc = MLPClassifier(hidden_layer_sizes=16, activation='logistic', solver='adam', momentum=0.2,learning_rate_init=0.001, random_state=1)
    mlpc = CalibratedClassifierCV(mlpc)

    # 模型训练、显示结果
    with st.spinner("Training, please wait..."):
        model_result = model_fit_score(mlpc, X_train, y_train)
    with col1:
        st.text("Training Result")
        msg = model_print(model_result, "MLPClassifier - Train")
        st.write(msg)
        # 训练画图
        # fig_train = plt_roc_auc([
        #     (model_result, 'MLPClassifier',),
        # ], 'Train ROC')
        # st.pyplot(fig_train)
    # 模型测试、显示结果
    with st.spinner("Testing, please wait..."):
        model_test_result = model_score(mlpc, X_test, y_test)
    with col2:
        st.text("Testing Result")
        msg = model_print(model_test_result, "MLPClassifier - Test")
        st.write(msg)
        # 测试画图
        # fig_test = plt_roc_auc([
        #     (model_test_result, 'MLPClassifier',),
        # ], 'Validate ROC')
        # st.pyplot(fig_test)


# 对生成的预测数据进行处理
def do_predict():
    global vars
    global mlpc

    # 处理生成的预测数据的输入
    pocd_predict = pd.DataFrame(data=[vars], columns=COL_INPUT)
    pocd_predict = do_base_preprocessing(pocd_predict, with_y=False)
    st.text("Preview for detail of this predict data")
    st.write(pocd_predict)
    pocd_predict = do_predict_preprocessing(pocd_predict)

    # 进行预测并输出
    pr = mlpc.predict(pocd_predict)
    pr = pr.astype(np.int)
    st.markdown(r"$\color{red}{MLPClassifier}$ $\color{red}{Predict}$ $\color{red}{result}$ $\color{red}{" + str(
        COL_Y[0]) + r"}$ $\color{red}{is}$ $\color{red}{" + str(pr[0]) + "}$")


if __name__ == "__main__":
    do_processing()
    setup_selectors()

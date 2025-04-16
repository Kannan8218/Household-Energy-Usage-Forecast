import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
class Household_Energy:
    def __init__(self):
        self.data = None
        self.numeric_features, self.categorical_features = None, None
        self.x, self.y = None, None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.x_train_scale, self.x_test_scale, self.y_train_scale, self.y_test_scale = None, None, None, None
        self.LR_scale, self.LR_model, self.LR_MAE, self.LR_MSE, self.LR_RMSE, self.LR_R2 = None, None, None, None, None, None
        self.knn_scale, self.knn_model, self.knn_MAE, self.knn_MSE, self.knn_RMSE, self.knn_R2 = None, None, None, None, None, None
        self.DT_scale, self.DT_model, self.DT_MAE, self.DT_MSE, self.DT_RMSE, self.DT_R2 = None, None, None, None, None, None
        self.RF_scale, self.RF_model, self.RF_MAE, self.RF_MSE, self.RF_RMSE, self.RF_R2 = None, None, None, None, None, None
        self.RE_scale, self.RE_model, self.RE_MAE, self.RE_MSE, self.RE_RMSE, self.RE_R2 = None, None, None, None, None, None
    def data_extract(self):
        st.title("Household power usage")
        dtype_spec = {'Date' : 'object', 'Time' : 'object', 'Global_active_power': 'float', 'Global_reactive_power': 'float',
                    'Voltage': 'float', 'Global_intensity': 'float', 'Sub_metering_1': 'float', 'Sub_metering_2': 'float',}
        try:
            self.data = pd.read_csv("household_power_consumption.txt", sep=";", dtype=dtype_spec, na_values="?")
        except Exception as e:
            st.error(f"❌ Error - data_extract() -> read household_power_consumption file try block\nERROR -> {e}")
            return False
        try :
            if len(self.data)>0:
                st.header(" 📋 Basic Information of Dataset")
                st.markdown(f"**Shape of the data:** `{self.data.shape}`")
                st.markdown(f"**Number of duplicate rows:** `{self.data.duplicated().sum()}`")
                st.markdown("**Columns in dataset:**")
                st.write(self.data.columns.tolist())
                st.markdown("**Data types of each column:**")
                data_types_df = pd.DataFrame({"Column": self.data.columns,"Data Type": [str(dtype) for dtype in self.data.dtypes]})
                st.dataframe(data_types_df)
                null_counts = self.data.isnull().sum()
                null_percent = (null_counts / len(self.data)) * 100
                null_summary = pd.DataFrame({"Missing Values Count": null_counts,"Missing Values(%)": null_percent.round(3)})
                null_summary = null_summary[null_summary["Missing Values Count"] > 0]
                st.markdown("**Missing Value Summary**")
                if not null_summary.empty:
                    st.dataframe(null_summary)
                else:
                    st.success("No missing values in the dataset.")
                st.markdown("**Data Info**")
                buffer = io.StringIO()
                self.data.info(buf=buffer)
                info_str = buffer.getvalue()
                st.text(info_str)
                st.markdown("**Descriptive Statistics**")
                st.dataframe(self.data.describe())
                st.markdown("**Preview of Dataset**")
                st.dataframe(self.data.head())
            else:
                st.error("data set is empty")
                return False
        except Exception as e:
            st.error(f"❌ Error - data_extract() -> try block\nERROR -> {e}")
            return False
        return True
    def data_clean(self):
        try:
            st.header("🧹 Data Clean process")
            a = self.data.shape[0]
            self.data = self.data.dropna()
            b = self.data.shape[0]
            row_summary = pd.DataFrame([{"Before Removing NaNs": a,"After Removing NaNs": b,"Total Rows Removed": a - b}])
            st.markdown("**Dataset Row Summary After Removing NaNs**")
            st.dataframe(row_summary)
            self.numeric_features = [feature for feature in self.data.columns if self.data[feature].dtype != 'O']
            self.categorical_features = [feature for feature in self.data.columns if self.data[feature].dtype == 'O']
            zero_stats = []
            for feature in self.numeric_features:
                num_zeros = (self.data[feature] == 0).sum()
                total_values = self.data[feature].count()
                non_zeros = total_values - num_zeros
                zero_stats.append({"Feature": feature, "Total Value Count": total_values,"Zero Value Count": num_zeros,"Non-Zero Value Count": non_zeros})
            zero_stats_df = pd.DataFrame(zero_stats)
            st.markdown("**Numeric Feature Summary**")
            st.dataframe(zero_stats_df)
            nan_stats = []
            for feature in self.categorical_features:
                nan_count = self.data[feature].isna().sum()
                nan_stats.append({"Feature": feature, "NaN Count": nan_count})
            st.markdown("**Categorical Feature Summary**")
            nan_df = pd.DataFrame(nan_stats)
            st.dataframe(nan_df)
            return True
        except Exception as e:
            st.error(f"❌ Error - data_clean() -> try block\nERROR -> {e}")
            return False
    def skew_analysis(self):
        st.header("📈 Skew Analysis")
        try:
            for feature in self.numeric_features:
                feature_data_copy = self.data[feature]
                fig, ax = plt.subplots(figsize=(10, 3))
                sns.kdeplot(x=feature_data_copy, fill=True, color='r', ax=ax)
                ax.set_xlabel(feature)
                ax.set_title(f'Distribution of {feature}', fontsize=14)
                skew_val = feature_data_copy.skew()
                if skew_val > 0.5:
                    skew_type = "Right-Skewed"
                elif skew_val < -0.5:
                    skew_type = "Left-Skewed"
                else:
                    skew_type = "Approximately Symmetric"
                fig.text(0.5, -0.1, f"Skewness: {skew_val:.2f} ({skew_type}) Before handle Outliers", ha="center", fontsize=12)
                fig.tight_layout()
                st.pyplot(fig)
        except Exception as e:
                st.error(f"❌ Error - skew_analysis() -> try block\nERROR -> {e}")
                return False
        return True
    def outlier_handle(self):
        try:
            st.header("📈 Outlier Detection and Handling")
            for feature in self.numeric_features:
                feature_data = self.data[feature]
                fig, ax = plt.subplots(figsize=(10, 3))
                sns.boxplot(x=feature_data, color='skyblue', ax=ax)
                ax.set_title(f'Boxplot of {feature} Before Capping')
                ax.set_xlabel(feature)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                Q1 = feature_data.quantile(0.25)
                Q3 = feature_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.data[feature] = np.where(self.data[feature] > upper_bound, upper_bound, self.data[feature])
                self.data[feature] = np.where(self.data[feature] < lower_bound, lower_bound, self.data[feature])
                capped_data = self.data[feature].dropna()
                outliers = capped_data[(capped_data < lower_bound) | (capped_data > upper_bound)]
                fig, ax = plt.subplots(figsize=(10, 3))
                sns.boxplot(x=capped_data, color='skyblue', ax=ax)
                ax.set_title(f'Boxplot of {feature} After Capping')
                ax.set_xlabel(feature)
                plt.tight_layout()
                st.pyplot(fig)
                st.write(f"**{feature}** — Outliers remaining after capping: `{len(outliers)}`")
                st.write(f"Lower Bound: `{lower_bound}`")
                st.write(f"Upper Bound: `{upper_bound}`")
                st.write(f"Max Value After Capping: `{self.data[feature].max()}`")
                st.write(f"Min Value After Capping: `{self.data[feature].min()}`")
                st.markdown("---")
        except Exception as e:
                st.error(f"❌ Error - outlier_handle() -> try block\nERROR -> {e}")
                return False
        return True
    def correlations(self):
        try:
            st.header("📈 correlations between columns")
            selected_df = self.data[self.numeric_features]
            correlation_matrix = selected_df.corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title("Correlation Heatmap")
            st.pyplot(plt)
        except Exception as e:
                st.error(f"❌ Error - correlations() -> try block\nERROR -> {e}")       
                return False
        return True      
    def model_setup(self):
        try:
        #set feature and target data
            self.x = self.data[['Global_reactive_power','Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2','Sub_metering_3']]
            self.y = self.data['Global_active_power']
            #split train and test data
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=62)
        except Exception as e:
                st.error(f"❌ Error - model_setup() -> try block\nERROR -> {e}")    
                return False
        return True
    def Linear_Regression(self):
        try:#here preprocessor the train & test dataset
            self.LR_scale = MinMaxScaler()
            self.x_train_scale = self.LR_scale.fit_transform(self.x_train)
            self.x_test_scale = self.LR_scale.transform(self.x_test)
        except Exception as e:
                st.error(f"❌ Error - Linear_Regression() -> scalling try block\nERROR -> {e}")   
                return False 
        try:#here train the model
            self.LR_model = LinearRegression()
            self.LR_model.fit(self.x_train_scale, self.y_train)
        except Exception as e:
                st.error(f"❌ Error - Linear_Regression() -> model train try block\nERROR -> {e}")     
                return False
        try:#find accuracy
            y_pred = self.LR_model.predict(self.x_test_scale)
            self.LR_R2 = round(r2_score(self.y_test, y_pred) * 100, 2)
            self.LR_MAE = metrics.mean_absolute_error(self.y_test, y_pred)
            self.LR_MSE = metrics.mean_squared_error(self.y_test, y_pred)
            self.LR_RMSE = np.sqrt(metrics.mean_squared_error(self.y_test, y_pred))
        except Exception as e:
                st.error(f"❌ Error - Linear_Regression() -> accuracy try block\nERROR -> {e}")  
                return False  
        st.markdown(f"**Linear Regression Model completed**")
        return True
    def knn_Regression(self):
        try:  # Preprocess the train & test dataset
            self.knn_scaler = StandardScaler()
            self.x_train_scale = self.knn_scaler.fit_transform(self.x_train)
            self.x_test_scale = self.knn_scaler.transform(self.x_test)
        except Exception as e:
            st.error(f"❌ Error - knn_Regression() -> scaling try block\nERROR -> {e}")   
            return False
        try:  # Train the model
            self.knn_model = KNeighborsRegressor(n_neighbors=6)
            self.knn_model.fit(self.x_train_scale, self.y_train)
        except Exception as e:
            st.error(f"❌ Error - knn_Regression() -> model train try block\nERROR -> {e}")  
            return False
        try:  # Find accuracy
            y_pred = self.knn_model.predict(self.x_test_scale)
            self.knn_MAE = metrics.mean_absolute_error(self.y_test, y_pred)
            self.knn_MSE = metrics.mean_squared_error(self.y_test, y_pred)
            self.knn_RMSE = np.sqrt(self.knn_MSE)
            self.knn_R2 = round(r2_score(self.y_test, y_pred) * 100, 2)
        except Exception as e:
            st.error(f"❌ Error - knn_Regression() -> accuracy try block\nERROR -> {e}")
            return False
        st.markdown(f"**KNN Regression Model completed**")
        return True
    def decision_tree(self):
        try:#here preprocessor the train & test dataset
            self.DT_scaler = RobustScaler()
            self.x_train_scale = self.DT_scaler.fit_transform(self.x_train)
            self.x_test_scale = self.DT_scaler.transform(self.x_test)
        except Exception as e:
                st.error(f"❌ Error - knn_Regression() -> scalling try block\nERROR -> {e}")   
                return False
        try:#here train the model
            self.DT_model = DecisionTreeRegressor()
            self.DT_model.fit(self.x_train_scale, self.y_train)
        except Exception as e:
                st.error(f"❌ Error - knn_Regression() -> model train try block\nERROR -> {e}")  
                return False
        try:#find accuracy
            y_pred = self.DT_model.predict(self.x_test_scale)
            self.DT_MAE = metrics.mean_absolute_error(self.y_test, y_pred)
            self.DT_MSE = metrics.mean_squared_error(self.y_test, y_pred)
            self.DT_RMSE = np.sqrt(metrics.mean_squared_error(self.y_test, y_pred))
            self.DT_R2 = round(r2_score(self.y_test, y_pred) * 100, 2)
        except Exception as e:
                st.error(f"❌ Error - knn_Regression() -> accuracy try block\nERROR -> {e}")  
                return False
        st.markdown(f"**Decision Tree Model completed**")
        return True
    def Random_forest(self):
        try:#here preprocessor the train & test dataset
            self.RF_scale = MaxAbsScaler()
            self.x_train_scale = self.RF_scale.fit_transform(self.x_train)
            self.x_test_scale = self.RF_scale.transform(self.x_test)
        except Exception as e:
                st.error(f"❌ Error - Random_forest() -> scalling try block\nERROR -> {e}")  
                return False
        try:#here train the model
            self.RF_model = RandomForestRegressor(n_estimators=10)
            self.RF_model.fit(self.x_train_scale, self.y_train)
        except Exception as e:
                st.error(f"❌ Error - Random_forest() -> model train try block\nERROR -> {e}")  
                return False
        try:#find accuracy
            y_pred = self.RF_model.predict(self.x_test_scale)
            r2 = round(r2_score(self.y_test, y_pred) * 100, 2)
            self.RF_MAE = metrics.mean_absolute_error(self.y_test, y_pred)
            self.RF_MSE = metrics.mean_squared_error(self.y_test, y_pred)
            self.RF_RMSE = np.sqrt(metrics.mean_squared_error(self.y_test, y_pred))
            self.RF_R2 = round(r2_score(self.y_test, y_pred) * 100, 2)
        except Exception as e:
                st.error(f"❌ Error - Random_forest() -> accuracy try block\nERROR -> {e}") 
                return False
        st.markdown(f"**Random Forest  Model completed**")
        return True
    def ridge(self):
        try:#here preprocessor the train & test dataset
            self.RE_scale = StandardScaler()
            self.x_train_scale = self.RE_scale.fit_transform(self.x_train)
            self.x_test_scale = self.RE_scale.transform(self.x_test)
        except Exception as e:
                st.error(f"❌ Error - ridge() -> scalling try block\nERROR -> {e}")  
                return False
        try:#here train the model
            self.RE_model = Ridge(alpha=1.0)  
            self.RE_model.fit(self.x_train_scale, self.y_train)
        except Exception as e:
                st.error(f"❌ Error - ridge() -> model train try block\nERROR -> {e}")  
                return False
        try:#find accuracy
            y_pred = self.RE_model.predict(self.x_test_scale)
            self.RE_MAE = metrics.mean_absolute_error(self.y_test, y_pred)
            self.RE_MSE = metrics.mean_squared_error(self.y_test, y_pred)
            self.RE_RMSE = np.sqrt(metrics.mean_squared_error(self.y_test, y_pred))
            self.RE_R2 = round(r2_score(self.y_test, y_pred) * 100, 2)
        except Exception as e:
                st.error(f"❌ Error - ridge() -> accuracy try block\nERROR -> {e}")  
                return False
        st.markdown(f"**Ridge Model completed**")
        return True
    def visual(self):
        try:
            st.subheader("📊 Model Evaluation Metrics")
            data = {"Model": ["Linear Regression", "Knn Regression", "Decision Tree", "Random Forest","Ridge Regression"],
                    "R2 value": [self.LR_R2, self.knn_R2, self.DT_R2, self.RF_R2, self.RE_R2]}
            df = pd.DataFrame(data)
            st.subheader("📋 R² Score Board")
            st.dataframe(df, use_container_width=True)
            model_scores = {'Linear Regression': self.LR_R2,'KNN Regression': self.knn_R2,'Decision Tree': self.DT_R2,
                            'Random Forest': self.RF_R2,'Ridge Regression': self.RE_R2}
            df_scores = pd.DataFrame(list(model_scores.items()), columns=['Model', 'R2_Score'])
            fig, ax = plt.subplots()
            ax.bar(df_scores['Model'], df_scores['R2_Score'], color='skyblue', edgecolor='black')
            ax.set_ylim(99, 100)
            ax.set_ylabel("R² Score")
            ax.set_title("Model R² Comparison")
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig)
            st.subheader("📋 Model Error Board")
            a = {"Model": ["Linear Regression", "Knn Regression", "Decision Tree", "Random Forest","Ridge Regression"],
                "MAE": [float(self.LR_MAE), float(self.knn_MAE), float(self.DT_MAE), float(self.RF_MAE), float(self.RE_MAE)],
                "MSE": [float(self.LR_MSE), float(self.knn_MSE), float(self.DT_MSE), float(self.RF_MSE), float(self.RE_MSE)],
                "RMSE": [float(self.LR_RMSE), float(self.knn_RMSE), float(self.DT_RMSE), float(self.RF_RMSE), float(self.RE_RMSE)]}
            df = pd.DataFrame(a)
            st.dataframe(df.style.format({"MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}"}), use_container_width=True)
        except Exception as e:
                st.error(f"❌ Error - visual() -> try block\nERROR -> {e}")  
                return False
if __name__ == '__main__':
    try:
        obj = Household_Energy()
    except Exception as e:
        print(f"❌ Error - data_extract() -> read household_power_consumption file try block\nERROR -> {e}")
        exit()
    if obj.data_extract():
        if obj.data_clean():
            if obj.skew_analysis():
                if obj.outlier_handle():
                    if obj.correlations():
                        if obj.model_setup():
                            if obj.Linear_Regression():
                                if obj.knn_Regression():
                                    if obj.decision_tree():
                                        if obj.Random_forest():
                                            if obj.ridge():
                                                obj.visual()
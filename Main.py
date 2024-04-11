import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

sns.set_theme(style="whitegrid")

def load_data(uploaded_file):
    """Load the uploaded CSV file into a DataFrame."""
    return pd.read_csv(uploaded_file)

def remove_columns(df, columns_to_remove):
    """Remove specified columns from the DataFrame."""
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(col, axis=1)
        else:
            st.warning(f"Column '{col}' not found in the dataset.")
    return df

def convert_df_to_csv(df):
    """Convert DataFrame to CSV for downloading."""
    return df.to_csv(index=False).encode('utf-8')

def plot_visualization(df, columns_to_plot, plot_type):
    """Generate and display the requested plot type."""
    fig, ax = plt.subplots()
    if len(columns_to_plot) < 1:
        st.warning("Please select at least one column for the visualization.")
        return

    if plot_type == 'Pairplot':
        if len(columns_to_plot) >= 2:
            sns.pairplot(df[columns_to_plot])
        else:
            st.warning("Please select at least two columns for the pairplot.")
            return
    elif plot_type == 'Barplot':
        if len(columns_to_plot) >= 2:
            sns.barplot(x=columns_to_plot[0], y=columns_to_plot[1], data=df, ax=ax)
        else:
            st.warning("Please select exactly two columns for the barplot.")
            return
    elif plot_type == 'Scatterplot':
        if len(columns_to_plot) >= 2:
            sns.scatterplot(x=columns_to_plot[0], y=columns_to_plot[1], data=df, ax=ax)
        else:
            st.warning("Please select at least two columns for the scatterplot.")
            return
    elif plot_type == 'Displot':
        sns.displot(df[columns_to_plot[0]], kde=True)
        plt.close()
        st.pyplot(fig)
        return
    elif plot_type == 'Boxplot':
        sns.boxplot(data=df[columns_to_plot], ax=ax)
    elif plot_type == 'Jointplot':
        sns.jointplot(x=columns_to_plot[0], y=columns_to_plot[1], data=df, kind="scatter")
        plt.close()
        st.pyplot(fig)
        return

    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    st.download_button(
        label="Download plot as PNG",
        data=buf,
        file_name=f"{plot_type}.png",
        mime="image/png"
    )

def show_dataset_description(df):
    """Show a more visually appealing detailed description of the dataset."""
    st.markdown("### Dataset Description")
    st.markdown(f"- **Rows:** {df.shape[0]}  \n- **Columns:** {df.shape[1]}  \n- **Total Elements:** {df.size}")

    dtype_summary = df.dtypes.value_counts().reset_index()
    dtype_summary.columns = ['Data Type', 'Count']
    st.table(dtype_summary)

    st.markdown("#### Descriptive Statistics")
    st.dataframe(df.describe())

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        st.markdown("#### Missing Values per Column")
        st.write(missing_values)

        st.markdown("#### Visual Representation of Missing Values")
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
        st.pyplot(fig)

def handle_null_values(df):
    """Provide options to handle null values in the DataFrame."""
    if df.isnull().sum().sum() > 0:
        st.warning("Warning: The dataset contains null values.")
        option = st.selectbox(
            "Choose how to handle null values:",
            ['Select', 'Remove rows with null values', 'Replace null values']
        )

        if option == 'Remove rows with null values':
            df = df.dropna()
            st.success("Rows with null values have been removed.")
        elif option == 'Replace null values':
            replace_with = st.text_input("Enter the value to replace null values with:")
            if replace_with.isnumeric():
                replace_with = float(replace_with) if '.' in replace_with else int(replace_with)
            if st.button("Replace"):
                df = df.fillna(replace_with)
                st.success("Null values have been replaced.")

    return df

def main():
    """Main function for the Streamlit app."""
    st.title("Data Science Assistant")

    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Data loaded successfully:", df.head())
        
        show_dataset_description(df)

        columns_to_remove = st.multiselect("Select columns to remove", df.columns)
        if st.button("Clean Data"):
            df_clean = remove_columns(df, columns_to_remove)
            df_clean = handle_null_values(df_clean)
            st.session_state.df_clean = df_clean
        st.write("Data after cleaning:", st.session_state.df_clean.head())

    if st.session_state.df_clean is not None:
        plot_type = st.selectbox("Select plot type", ['Pairplot', 'Barplot', 'Scatterplot', 'Displot', 'Boxplot', 'Jointplot'])
        columns_to_plot = st.multiselect("Select columns to plot", st.session_state.df_clean.columns)
        
        if st.button("Generate Plot"):
            plot_visualization(st.session_state.df_clean, columns_to_plot, plot_type)

        csv = convert_df_to_csv(st.session_state.df_clean)
        st.download_button(
            label="Download cleaned data as CSV",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()

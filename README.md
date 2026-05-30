[pro.py](https://github.com/user-attachments/files/28422553/pro.py)
import sys
import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel, QDialog, QScrollArea
from PyQt5.QtCore import Qt
from collections import Counter
from PyQt5.QtWidgets import QMessageBox, QTableWidget, QTableWidgetItem
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim import corpora, models
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

class CSVFileAnalyzer(QMainWindow):
    def _init_(self):
        super()._init_()

        self.setWindowTitle("CSV File Analyzer")
        self.setGeometry(100, 100, 400, 400)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        self.load_button = QPushButton("Load CSV")
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)

        self.save_button = QPushButton("Save All as CSV Files")
        self.save_button.clicked.connect(self.save_csv_files)
        layout.addWidget(self.save_button)

        # Add the "Ticket Description" button to the layout
        self.ticket_description_button = QPushButton("Ticket Description")
        self.ticket_description_button.clicked.connect(self.analyze_ticket_descriptions)
        layout.addWidget(self.ticket_description_button)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        self.button_widgets = []
        self.data = None
        self.product_purchased_label = QLabel("")
        layout.addWidget(self.product_purchased_label)

        self.central_widget.setLayout(layout)

        self.ticket_type_data = {}  # Dictionary to store data frames for each Ticket Type

    def analyze_ticket_descriptions(self):
        if self.data is not None:
            ticket_descriptions = self.data["Ticket Description"]

            # Analyze Ticket Descriptions to extract keywords using LDA
            keywords = self.extract_key_phrases_lda(self.data, "Ticket Description")

            # Create buttons for unique keywords
            self.create_keyword_buttons(keywords)

    def extract_key_phrases_lda(self, data, text_column, num_topics=5, num_words=5):
        # Preprocess the text
        data[text_column] = data[text_column].str.replace(r'{product_purchased}', '')  # Remove "product_purchased"
        data[text_column] = data[text_column].str.replace(r'[.,!{}[\]/\\]', '')  # Remove specified punctuation

        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        data['Tokenized_Description'] = data[text_column].apply(lambda x: [word for word in word_tokenize(x) if word.lower() not in stop_words])

        # Create a dictionary and a document-term matrix
        dictionary = corpora.Dictionary(data['Tokenized_Description'])
        corpus = [dictionary.doc2bow(text) for text in data['Tokenized_Description']]

        # Build the LDA model
        lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

        # Extract keywords from LDA topics
        topics = lda_model.show_topics(formatted=False, num_words=num_words)
        keywords = []

        for topic in topics:
            topic_keywords = [word for word, prob in topic[1]]
            keywords.extend(topic_keywords)

        return keywords

    def create_keyword_buttons(self, keywords):
        keyword_dialog = QDialog(self)
        keyword_dialog.setWindowTitle("Keyword Analysis")
        keyword_dialog.setGeometry(300, 300, 400, 400)

        layout = QVBoxLayout(keyword_dialog)
        layout.addWidget(QLabel("Keywords:"))

        scroll_area = QScrollArea(keyword_dialog)
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget(scroll_area)
        button_layout = QVBoxLayout(content_widget)

        for keyword in keywords:
            button = QPushButton(keyword)
            self.button_widgets.append(button)
            button_layout.addWidget(button)
            button.clicked.connect(lambda state, keyword=keyword: self.show_ticket_description_by_keyword(keyword))

        content_widget.setLayout(button_layout)
        scroll_area.setWidget(content_widget)

        layout.addWidget(scroll_area)
        keyword_dialog.setLayout(layout)
        keyword_dialog.exec()

    def show_ticket_description_by_keyword(self, keyword):
        if self.data is not None:
            matching_descriptions = self.data[self.data["Ticket Description"].str.contains(keyword, case=False)]

        total_descriptions = len(self.data)
        matching_count = len(matching_descriptions)
        percentage = (matching_count / total_descriptions) * 100

        message = f"Keyword: {keyword}\n"
        message += f"Matching Descriptions: {matching_count} out of {total_descriptions} ({percentage:.2f}%)\n\n"

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Matching Descriptions")
        msg.exec()

    def load_csv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (.csv);;All Files ()", options=options)

        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                unique_ticket_types = self.data["Ticket Type"].unique()

                for ticket_type in unique_ticket_types:
                    button = QPushButton(ticket_type)
                    self.button_widgets.append(button)
                    layout = self.central_widget.layout()
                    layout.addWidget(button)
                    button.clicked.connect(lambda state, tt=ticket_type: self.show_product_purchased(tt))

                self.result_label.setText("Loaded unique values under the 'Ticket Type' column.")

                for ticket_type in unique_ticket_types:
                    self.ticket_type_data[ticket_type] = self.data[self.data["Ticket Type"] == ticket_type]
            except Exception as e:
                self.result_label.setText("An error occurred: " + str(e))
        else:
            self.result_label.setText("No file selected.")

    def show_product_purchased(self, ticket_type):
        if self.data is not None:
            selected_data = self.ticket_type_data.get(ticket_type)

            if selected_data is not None:
                unique_products = selected_data["Product Purchased"].unique().tolist()
                product_window = ProductPurchasedWindow(unique_products, selected_data, self)
                product_window.exec()

    def save_csv_files(self):
        if self.data is not None:
            output_folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save CSV Files")

            if output_folder:
                unique_ticket_types = self.data["Ticket Type"].unique()
                for ticket_type in unique_ticket_types:
                    filename = os.path.join(output_folder, f"{ticket_type}.csv")
                    subset = self.data[self.data["Ticket Type"] == ticket_type]
                    subset.to_csv(filename, index=False)

                self.result_label.setText(f"Saved {len(unique_ticket_types)} CSV files in '{output_folder}'.")

class ProductPurchasedWindow(QDialog):
    def _init_(self, products, data, main_window):
        super()._init_()

        self.setWindowTitle("Product Purchased")
        self.setGeometry(200, 200, 400, 400)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        content = QWidget(scroll_area)
        layout = QVBoxLayout(content)

        self.product_label = QLabel("Unique 'Product Purchased' values:")
        layout.addWidget(self.product_label)

        self.button_widgets = []

        for product in products:
            button = QPushButton(product)
            self.button_widgets.append(button)
            layout.addWidget(button)
            button.clicked.connect(lambda state, product=product: self.save_product_data(product, data, main_window))

        back_button = QPushButton("Back to Main Screen")
        layout.addWidget(back_button)
        back_button.clicked.connect(self.close)

        content.setLayout(layout)
        scroll_area.setWidget(content)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

        self.main_window = main_window

    def save_product_data(self, product, data, main_window):
        if data is not None:
            subset = data[data["Product Purchased"] == product]
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, f"Save {product} Data as Excel", "", "Excel Files (.xlsx);;All Files ()", options=options)

            if file_path:
                try:
                    if not file_path.endswith('.xlsx'):
                        file_path += '.xlsx'
                    subset.to_excel(file_path, index=False)

                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Question)
                    msg.setText(f"Do you want to continue studying {product}?")
                    msg.setWindowTitle("Continue Studying?")
                    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    result = msg.exec()

                    if result == QMessageBox.Yes:
                        self.show_column_names(subset)
                    else:
                        self.accept()
                except Exception as e:
                    print(f"An error occurred while saving data for {product}: {str(e)}")
            else:
                print(f"No file selected for saving {product} data.")
                
    def show_column_names(self, data):
        column_names = list(data.columns)

        column_names_dialog = QDialog(self)
        column_names_dialog.setWindowTitle("Column Names")
        column_names_dialog.setGeometry(300, 300, 400, 400)

        layout = QVBoxLayout(column_names_dialog)

        for column_name in column_names:
            if column_name != "Ticket Description":
                button = QPushButton(column_name)
                layout.addWidget(button)
                button.clicked.connect(lambda state, column_name=column_name: self.show_unique_values(data, column_name))

        back_button = QPushButton("Back to Main Screen")
        layout.addWidget(back_button)
        back_button.clicked.connect(column_names_dialog.accept)

        column_names_dialog.setLayout(layout)
        column_names_dialog.exec()

    def show_unique_values(self, data, column_name):
        if column_name == "Ticket Description":
            pass
        else:
            unique_values = data[column_name].unique()
            total_count = len(data)

            unique_values_counts = data[column_name].value_counts()

            unique_values_dialog = QDialog(self)
            unique_values_dialog.setWindowTitle(f"Unique Values for '{column_name}'")
            unique_values_dialog.setGeometry(400, 400, 600, 400)

            scroll_area = QScrollArea(unique_values_dialog)
            scroll_area.setWidgetResizable(True)

            content = QWidget(scroll_area)
            layout = QVBoxLayout(content)

            max_width = max(len(str(value)) for value in unique_values)

            for value in unique_values:
                count = unique_values_counts[value] if value in unique_values_counts else 0
                percentage = (count / total_count) * 100

                value_str = str(value).rjust(max_width)
                count_str = str(count).rjust(8)
                percentage_str = f"{percentage:.2f}%".rjust(8)

                label_text = f"{value_str} ({count_str} / {total_count} - {percentage_str})"
                value_button = QPushButton(label_text)
                layout.addWidget(value_button)
                value_button.clicked.connect(lambda state, value=value: self.show_customer_info(data, column_name, value))

            back_button = QPushButton("Back to Values")
            layout.addWidget(back_button)
            back_button.clicked.connect(unique_values_dialog.accept)

            content.setLayout(layout)
            scroll_area.setWidget(content)

            unique_values_dialog.setLayout(QVBoxLayout())
            unique_values_dialog.layout().addWidget(scroll_area)
            unique_values_dialog.exec()

    def show_customer_info(self, data, column_name, value):
        customer_info_dialog = QDialog(self)
        customer_info_dialog.setWindowTitle(f"Customer Information for '{column_name}': {value}")
        customer_info_dialog.setGeometry(400, 400, 800, 600)

        layout = QVBoxLayout(customer_info_dialog)

        filtered_data = data[data[column_name] == value]

        table = QTableWidget()
        table.setColumnCount(len(filtered_data.columns))
        table.setRowCount(len(filtered_data))
        table.setHorizontalHeaderLabels(filtered_data.columns)

        for row in range(len(filtered_data)):
            for col in range(len(filtered_data.columns)):
                item = QTableWidgetItem(str(filtered_data.iat[row, col]))
                table.setItem(row, col, item)

        layout.addWidget(table)

        back_button = QPushButton("Back to Values")
        layout.addWidget(back_button)
        back_button.clicked.connect(customer_info_dialog.accept)

        customer_info_dialog.setLayout(layout)
        customer_info_dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSVFileAnalyzer()
    window.show()
    sys.exit(app.exec())

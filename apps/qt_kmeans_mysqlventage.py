import sys
import argparse
import pandas as pd
from random import randrange
import numpy as np

from PySide2.QtCore import QAbstractTableModel, QModelIndex, QRect, Qt
from PySide2.QtGui import QColor, QPainter
from PySide2.QtWidgets import (QApplication, QGridLayout, QHeaderView,
    QTableView, QWidget, QVBoxLayout)
from PySide2.QtCharts import QtCharts
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from teradataml.analytics.mle.KMeans import KMeans
from teradataml.dataframe.dataframe import DataFrame
from teradataml.data.load_example_data import load_example_data
from teradataml import create_context

class CustomTableModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class MatplotlibWidget(QWidget):
    def __init__(self, data, parent=None):
        super().__init__(parent)

        fig = Figure(figsize=(15, 10))
        self.canvas = FigureCanvas(fig)
        self.navbar = NavigationToolbar(self.canvas, self)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.navbar)
        self.layout.addWidget(self.canvas)

        sno_to_state = dict(zip(data['sno'], data['state']))

        train_data = data.drop(['state'], axis=1)
        kMeans_output = KMeans(data=train_data, centers=2, data_sequence_column=['sno'])
        df1 = data.join(kMeans_output.clustered_output, how='inner', on=['sno'])
        df3 = df1[['sno', 'urban_pop', 'murder', 'clusterid']]
        numpy_df = df3.values

        ax = fig.add_subplot(111)
        ax.set_title('k-means clustering')
        ax.set_xlabel('urban_pop')
        ax.set_ylabel('murder')

        ax.scatter(numpy_df[:,1], numpy_df[:,2], c=numpy_df[:,3])

        for ind, value in enumerate(numpy_df[:, 0]):
            ax.text(numpy_df[ind,1], numpy_df[ind,2], sno_to_state[int(value)], size=10)

        self.setLayout(self.layout)


class TableWidget(QWidget):
    def __init__(self, df):
        QWidget.__init__(self)

        model = CustomTableModel(df.drop(['sno'], axis=1))
        table_view = QTableView()
        table_view.setModel(model)
        table_view.setMinimumSize(518, 600)

        plt = MatplotlibWidget(df)

        main_layout = QGridLayout()
        main_layout.addWidget(table_view, 1, 0)
        main_layout.addWidget(plt, 1, 1)
        self.setLayout(main_layout)


def read_data(fname):
    return pd.read_csv(fname)


if __name__ == "__main__":
    # options = argparse.ArgumentParser()
    # options.add_argument("-f", "--file", type=str, required=True)
    # args = options.parse_args()
    create_context(host = "localhost", username="root", password = "mysqlpass", temp_database_name='menagerie')

    load_example_data("kmeans", "kmeans_us_arrests_data")

    ## Creating TeradataML dataframes
    df_train = DataFrame('kmeans_us_arrests_data')

    df = df_train.to_pandas()

    # Qt Application
    app = QApplication(sys.argv)

    w = TableWidget(df)
    w.show()

    sys.exit(app.exec_())

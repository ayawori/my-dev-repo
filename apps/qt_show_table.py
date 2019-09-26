import sys
import argparse
import pandas as pd
from random import randrange

from PySide2.QtCore import QAbstractTableModel, QModelIndex, QRect, Qt
from PySide2.QtGui import QColor, QPainter
from PySide2.QtWidgets import (QApplication, QGridLayout, QHeaderView,
    QTableView, QWidget)
from PySide2.QtCharts import QtCharts


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


class TableWidget(QWidget):
    def __init__(self, data):
        QWidget.__init__(self)

        model = CustomTableModel(data)
        table_view = QTableView()
        table_view.setModel(model)
        table_view.setMinimumSize(640, 480)

        main_layout = QGridLayout()
        main_layout.addWidget(table_view)
        self.setLayout(main_layout)


def read_data(fname):
    return pd.read_csv(fname)


if __name__ == "__main__":
    # options = argparse.ArgumentParser()
    # options.add_argument("-f", "--file", type=str, required=True)
    # args = options.parse_args()
    df = read_data('kmeans_us_arrests_data.csv')

    # Qt Application
    app = QApplication(sys.argv)

    w = TableWidget(df)
    w.show()

    sys.exit(app.exec_())

                        # Imports and data preparation

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle  # Na uloženie modelu
from flask import Flask, request, jsonify  # Flask pre nasadenie API

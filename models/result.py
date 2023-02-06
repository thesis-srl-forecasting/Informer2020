import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ProcessedResult():
    def __init__(self, preds, trues, train_scaler):
        self.scaler = train_scaler
        self.pred_len = preds.shape[0]
        self.pred_raw = self.convert_seq(preds, inverse=False)
        self.true_raw = self.convert_seq(trues, inverse=False)
        self.pred = self.convert_seq(preds, inverse=True)
        self.true = self.convert_seq(trues, inverse=True)
        self.pred_naive = self.true.shift(1)
    
    def convert_seq(self, seq_raw, inverse=True):
        if inverse: 
            seq = self.scaler.inverse_transform(seq_raw)
        else: seq = seq_raw
        array = seq.squeeze()
        array = np.array([np.concatenate([np.repeat(np.nan, i), array[i], np.repeat(np.nan, self.pred_len-i)]) for i in np.arange(self.pred_len)])
        df = pd.DataFrame(array.transpose())
        return df.mean(axis=1)

    def plot_pred_vs_true(self, pred):
        fig = plt.figure(figsize=(12,6))
        plt.plot(self.true, label='True')
        plt.plot(pred, label ='Predict')
        plt.annotate(f'Predicted revenue: {self.predict_revenue(pred)}€', 
                     xy=(0.05, 0.9), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=1))
        plt.legend()
        # for idx in df_rev.index:
        #     plt.vlines(idx, df_rev.loc[idx]['revenue'],0, label ='Revenue', linestyle='--') 
        plt.close()
        return(fig)
    
    def predict_revenue(self, pred):
        return np.nansum(np.where(pred > self.true, 0, pred)).round(2)
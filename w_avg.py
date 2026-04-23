import pandas as pd
import numpy as np

# ১. আপনার ডেটা এখানে ইনপুট দিন
# দ্রষ্টব্য: আপনার কনফিউশন ম্যাট্রিক্সের সারি (Row) যোগ করে 'Support' এর মান বসাবেন।
data = {
    'Class_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    
    # আপনার বার চার্ট বা হিটম্যাপ থেকে পাওয়া স্কোর (দশমিকে)
    'Precision': [0.91, 0.87, 0.99, 0.92, 0.89, 0.90, 0.90, 0.97, 1.00, 0.95, 0.98, 0.99],
    'Recall':    [0.97, 0.98, 0.96, 0.97, 0.96, 0.85, 0.95, 0.86, 1.00, 0.91, 0.95, 0.84],
    'F1_Score':  [0.94, 0.92, 0.97, 0.94, 0.92, 0.88, 0.92, 0.91, 1.00, 0.93, 0.96, 0.91],
    
    # কনফিউশন ম্যাট্রিক্সের সারি যোগ করে পাওয়া আসল সংখ্যা (উদাহরন হিসেবে কিছু সংখ্যা দিলাম)
    # আপনি আপনার ম্যাট্রিক্সের প্রতিটি রো যোগ করে সঠিক সংখ্যা বসাবেন
    'Support':   [189, 214, 295, 196, 295, 129, 149, 294, 268, 278, 275, 104] 
}

# ডেটাফ্রেমে রূপান্তর
df = pd.DataFrame(data)

# ২. ওয়েটেড এভারেজ ক্যালকুলেশন ফাংশন
def calculate_weighted_avg(metric_name):
    # (Metric * Support) / Total Support
    weighted_score = np.average(df[metric_name], weights=df['Support'])
    return weighted_score

# ৩. ফলাফল বের করা
w_precision = calculate_weighted_avg('Precision')
w_recall = calculate_weighted_avg('Recall')
w_f1 = calculate_weighted_avg('F1_Score')

print("=== Weighted Averages ===")
print(f"Weighted Precision: {w_precision:.4f} (or {w_precision*100:.2f}%)")
print(f"Weighted Recall:    {w_recall:.4f} (or {w_recall*100:.2f}%)")
print(f"Weighted F1-Score:  {w_f1:.4f} (or {w_f1*100:.2f}%)")
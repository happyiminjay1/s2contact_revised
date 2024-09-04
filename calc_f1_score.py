precision_hand= 0.4168 
recall_hand= 0.8675 
precision_obj= 0.4241 
recall_obj= 0.7245

f1_score_hand = 2 * (precision_hand * recall_hand) / (precision_hand + recall_hand)
f1_score_obj = 2 * (precision_obj * recall_obj) / (precision_obj + recall_obj)

print('{:.3f}'.format(f1_score_hand))
print('{:.3f}'.format(f1_score_obj))
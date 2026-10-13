\# FedAvg pseudocode (PS-003)



initialise global model W

for round = 1 ... R:

&#x20;   for each client k (simulated junction):

&#x20;       W\_k = copy of W                      # client downloads global model

&#x20;       train W\_k on client k's local data   # raw data never leaves the client

&#x20;       send W\_k to server                   # upload weights only

&#x20;   W = sum over k of (n\_k / n\_total) \* W\_k  # FedAvg: weighted average

&#x20;   bytes\_this\_round = model\_size \* num\_clients \* 2   # down + up

&#x20;   evaluate W on the chronological test set -> MAE, RMSE

repeat for IID and non-IID client splits, with 3 random seeds


# Dataset

We use the IBM Telco Customer Churn sample referenced in the course notebook:

- [Kaggle dataset page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM source CSV](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)

The raw file has 7,043 rows and 21 columns. Each row describes one customer.
`Churn` is the outcome (`Yes` or `No`); `customerID` is an identifier and is
excluded from model inputs. The remaining fields describe demographics,
services, contract terms, tenure and charges.

## Obtain the data

From the repository root, after installing the package:

```bash
python -m telco_churn.data
```

The downloader writes `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` and checks:

```text
SHA-256: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

Alternatively, download the CSV from Kaggle and save it under the same name.
The raw CSV is excluded from Git. Dataset rights remain with the source;
the repository's MIT license does not relicense the dataset.

## Cleaning and scope

`TotalCharges` contains 11 blank strings. Converting it to numeric reveals
these as missing values; all 11 rows have zero tenure. We omit those rows
for the classification experiment, leaving 7,032 customers (5,163 stayed,
1,869 churned). This choice follows the complete-charge subset used in the
coursework, so the reported test results do not cover customers at zero tenure.
There are no exact duplicate rows or duplicate customer identifiers in the
verified sample.

The data describe a customer snapshot, not a sequence of billing records.
They contain no observed retention treatments or treatment outcomes.

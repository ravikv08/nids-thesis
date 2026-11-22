# Data-Mining Based Network Intrusion Detection System (NIDS)

**An ensemble approach combining C4.5 decision trees and MultiBoosting for anomaly-based intrusion detection**

V. Ravi Kumar · M.Tech, Computer Science & Engineering · Osmania University · 2010–11
Advisor: M. Swamy Das, Associate Professor, Dept. of CSE

*Summary of the original thesis. The code in this repository is a 2026 Python reproduction; the original system
was built in Java with Weka.*

## Abstract

Firewalls, authentication and signature-based intrusion detection need constant manual updates and miss attacks
they have not seen before. This thesis builds a data-mining based Network Intrusion Detection System (NIDS) that
moves from signature matching to anomaly detection, using a three-stage pipeline:

1. **Per-class feature selection** using information gain, to improve precision for low-frequency attack types.
2. **Binary C4.5 decision trees**, one per attack class.
3. A **MultiBoosting ensemble** that combines the classifiers by weighted majority vote, reducing bias and
   variance at the same time.

Evaluated on the DARPA **KDD Cup 1999** dataset, the system reached a **93.81% detection rate**, an improvement
over standalone AdaBoost or bagging, and detects four attack families: Denial of Service (DoS), Probe,
User-to-Root (U2R) and Remote-to-Local (R2L).

## Method

- **Data.** KDD Cup 99 connection records in ARFF format (duration, protocol, service, flag, byte counts,
  login flags, connection counts and rate features), each labelled `normal`, `probe`, `DoS`, `U2R` or `R2L`.
- **Feature selection.** Removing uninformative features improves accuracy and speeds up training. Features are
  ranked by information gain separately for each class.
- **C4.5.** Builds a decision tree by repeatedly splitting on the attribute with the highest information gain
  until every branch holds a single class (Quinlan, 1993).
- **MultiBoosting.** Combines AdaBoost (strong bias and variance reduction) with wagging (stronger variance
  reduction). The boosting rounds are split into sub-committees; at the start of each one the training weights
  are reset to random continuous-Poisson weights. With C4.5 as the base learner, MultiBoosting produces
  committees with lower error than either AdaBoost or wagging (Webb, 2000).
- **Training loop.** For each trial *t = 1..T*: the multibooster generates a weighted training set; binary C4.5
  classifiers are trained per class on that class's selected features; the ensemble's error sets the next
  round's weights. After *T* trials the final committee is used for detection.

## System

Packets were captured live with WinPcap, converted into the same attributes as the dataset, classified by the
C4.5 + MultiBoosting committee, and reported to an authenticated admin web page. DoS was the most frequent
attack type observed.

## Result

**93.81% detection rate** (gain-ratio criterion) on KDD Cup 99.

## Future work (as written in 2011)

Extend the analysis to current network data, such as the PREDICT dataset and real network traffic. The NSL-KDD
experiments in this repository are a step in that direction (see the [README](../README.md)).

## References

1. K. Ghosh, A. Schwartzbard, M. Schatz. Learning program behavior profiles for intrusion detection. USENIX Workshop on Intrusion Detection and Network Monitoring, 1999.
2. W. Lee, S. J. Stolfo. Data mining approaches for intrusion detection. USENIX Security Symposium, 1998.
3. W. Lee, S. J. Stolfo, K. W. Mok. A data mining framework for building intrusion detection models. IEEE Symposium on Security and Privacy, 1999.
4. S. Kumar. Classification and detection of computer intrusions. Ph.D. thesis, Purdue University, 1995.
5. J. R. Quinlan. C4.5: Programs for Machine Learning. Morgan Kaufmann, 1993.
6. The UCI KDD Archive. KDD Cup 1999 data.
7. T. Brugger. KDD Cup '99 dataset (network intrusion) considered harmful. KDnuggets, 2007.
8. S. Chebrolu, A. Abraham, J. P. Thomas. Feature deduction and ensemble design of intrusion detection systems. Computers & Security 24(4), 2005.
9. G. I. Webb. MultiBoosting: a technique for combining boosting and wagging. Machine Learning 40, 2000.

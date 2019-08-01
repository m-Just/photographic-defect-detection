This folder contains labels of multiple datasets, which are:
1. Flickr dataset (flickr)
  - `train/defect_training_gt_new.csv`: labels of training set
  - `test/defect_testing_gt_new.csv`: labels of testing set
  - `flickr_labels_ours.csv`: labels of testing set (annotated by our team)
  - `flickr_labels_anno_team.csv`: labels of testing set (annotated by the annotation team)
2. HDR+ dataset (hdr)
  - `hdr_train/hdr_labels.csv`: labels without Bad White Balance and Bad Composition labels.
  - `hdr_train/hdr_labels_025_comp_pred.csv`: labels with Bad Composition labels predicted by v0.2.5 model.
  - `hdr_train/hdr_labels_025_wb_comp_pred.csv`: labels with Bad White Balance and Bad Composition labels predicted by v0.2.5 model.
  - `hdr_train/hdr_labels_0626.csv`: newly annotated labels without Bad White Balance and Bad Composition labels.
  - `hdr_train/hdr_labels_0626_025_comp_pred.csv`: newly annotated labels with Bad Composition labels predicted by v0.2.5 model.
  - `hdr_train/hdr_labels_0626_025_wb_comp_pred.csv`: newly annotated labels with Bad White Balance and Bad Composition labels predicted by v0.2.5 model.
  - `hdr_train/hdr_labels_0626_std.csv`: the standard deviation of the annotations of each image of the newly annotated labels.
3. AVA, Adobe5K, HDR+ mixed dataset (mixed)
  - `mixed_test.csv`: deprecated version of labels of testing set
  - `mixed_train.csv`: deprecated version of labels of training set
  - `average`: currently using labels

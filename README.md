# BENDR

_BErt-like Neurophysiological Data Representation_

![A picture of Bender from Futurama][logo]

This repository contains the source code for reproducing, or extending the BERT-like self-supervision pre-training for EEG data from the article:

[BENDR: using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data](https://arxiv.org/pdf/2101.12037.pdf)

To run these scripts, you will need to use the [DN3](https://dn3.readthedocs.io/en/latest/) project. We will try to keep this updated so that it works with the latest DN3 release. If you are just looking for the BENDR model, and don't need to reproduce the article results _per se_, BENDR will be (or maybe already is if I forgot to update it here) integrated into DN3, in which case I would start there.

Currently, we recommend version [0.2](https://github.com/SPOClab-ca/dn3/tree/v0.2-alpha). Feel free to open an issue if you are having any trouble.

More extensive instructions are upcoming, but in essence you will need to either:

    a)  Download the TUEG dataset and pre-train new encoder and contextualizer weights, _or_
    b)  Use the [pre-trained model weights](https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha)

Once you have a pre-trained model:

    1) Add the paths of the pre-trained weights to configs/downstream.yml
    2) Edit paths to local copies of your datasets in configs/downstream_datasets.yml
    3) Run downstream.sh

#

# Improvements

[logo]: BENDR-jacking-on.gif "Bender Jacking-on"

This is a fork from the original [repository](https://github.com/SPOClab-ca/BENDR). The weights of the oretrained models are available in the original repository.

The processing of the DREAMER and SEED datasets. Two datasets for Emotion Recognition.

A method to reduce the number of EEG channels is added:

- in the dn3 library, in dn3.transforms.instance, you need to add the following class:

  ```
  class ToSEED(InstanceTransform):

      EEG_20_div = [
              'FP1', 'FP2',
          'F7', 'F3', 'FZ', 'F4', 'F8',
          'T7', 'C3', 'CZ', 'C4', 'T8',
          'T5', 'P3', 'PZ', 'P4', 'T6',
                  'O1', 'O2'
      ]

      EEG_SEED_div = list(map(str.upper, ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
                      'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'T5', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'T6',
                      'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']))

      def __init__(self, only_trial_data=True, include_scale_ch=True, include_ref_chs=False, include_extra_chs=True):
          """
          Transforms incoming Deep1010 data into exclusively the non-0 channel set.
          """
          super(ToSEED, self).__init__(only_trial_data=only_trial_data)
          self._inds_SEED_div = [DEEP_1010_CHS_LISTING.index(ch) for ch in self.EEG_SEED_div]
          if include_ref_chs:
              self._inds_SEED_div.append([DEEP_1010_CHS_LISTING.index(ch) for ch in ['A1', 'A2']])
          if include_extra_chs:
              self._inds_SEED_div += EXTRA_INDS[:-1]
          if include_scale_ch:
              self._inds_SEED_div.append(SCALE_IND)

      def new_channels(self, old_channels):
          return old_channels[self._inds_SEED_div]

      def call(self, x):
          x = list(x)
          for i in range(len(x)):
          # Assume every tensor that has deep1010 length should be modified
              if len(x[i].shape) > 0 and x[i].shape[0] == len(DEEP_1010_CHS_LISTING):
                  x[i] = x[i][self._inds_SEED_div, ...]
          return x
  ```

I reported an [issue](https://github.com/SPOClab-ca/dn3/issues/84) in the dn3 library that I reported but was not resolved.

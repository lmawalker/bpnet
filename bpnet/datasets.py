import pandas as pd
import numpy as np
from copy import deepcopy
import json
from pathlib import Path
from kipoi.data import Dataset
# try:
# import torch
# from bpnet.data import Dataset
# torch.multiprocessing.set_sharing_strategy('file_system')
# except:
#     print("PyTorch not installed. Using Dataset from kipoi.data")
#    from kipoi.data import Dataset

from kipoi.metadata import GenomicRanges
from bpnet.utils import to_list
from bpnet.dataspecs import DataSpec
from bpnet.preproc import bin_counts, keep_interval, moving_average, IntervalAugmentor
from bpnet.extractors import _chrom_sizes, _chrom_names
from concise.utils.helper import get_from_module
from tqdm import tqdm
from concise.preprocessing import encodeDNA
from random import Random
import joblib
from bpnet.preproc import resize_interval
from genomelake.extractors import FastaExtractor, BigwigExtractor, ArrayExtractor, BaseExtractor
from kipoi_utils.data_utils import get_dataset_item
from kipoiseq.dataloaders.sequence import BedDataset
import gin
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class LaceyFastaExtractor(BaseExtractor):
    def __init__(self, datafile, **kwargs):
        from pysam import FastaFile
        super(LaceyFastaExtractor, self).__init__(datafile, **kwargs)
        self.fasta = FastaFile(self._datafile)

    def _extract(self, intervals, out, **kwargs):
        for index, interval in enumerate(intervals):
            seq = self.fasta.fetch(str(interval.chrom), interval.start,
                                       interval.stop)
            row_idx = 0
            for base in seq:
                if base == 'A' or base == 'a':
                    out[index,row_idx, 0] = 1
                    out[index,row_idx, 2] = 1
                if base == 'C' or base == 'c':
                    out[index,row_idx, 1] = 1
                if base == 'G' or base == 'g':
                    out[index,row_idx, 1] = 1
                    out[index,row_idx, 3] = 1
                if base == 'T' or base == 't':
                    out[index,row_idx, 0] = 1
                if base == 'N' or base == 'n':
                    out[index,row_idx, 0] = 0.25
                    out[index,row_idx, 1] = 0.25
                    out[index,row_idx, 2] = 0.25
                    out[index,row_idx, 3] = 0.25

                row_idx += 1

        return out

    @staticmethod
    def _get_output_shape(num_intervals, width):
        NUM_SEQ_CHARS = 4
        return (num_intervals, width, NUM_SEQ_CHARS)

class LaceyFastaExtractorAG(BaseExtractor):
    def __init__(self, datafile, **kwargs):
        from pysam import FastaFile
        super(LaceyFastaExtractorAG, self).__init__(datafile, **kwargs)
        self.fasta = FastaFile(self._datafile)

    def _extract(self, intervals, out, **kwargs):
        for index, interval in enumerate(intervals):
            seq = self.fasta.fetch(str(interval.chrom), interval.start,
                                       interval.stop)
            row_idx = 0
            for base in seq:
                if base == 'A' or base == 'a':
                    out[index,row_idx, 0] = 1
                    out[index,row_idx, 2] = 1
                if base == 'C' or base == 'c':
                    out[index,row_idx, 1] = 1
                    out[index, row_idx, 3] = 1
                if base == 'G' or base == 'g':
                    out[index,row_idx, 0] = 1
                if base == 'T' or base == 't':
                    out[index,row_idx, 1] = 1
                if base == 'N' or base == 'n':
                    out[index,row_idx, 0] = 0.25
                    out[index,row_idx, 1] = 0.25
                    out[index,row_idx, 2] = 0.25
                    out[index,row_idx, 3] = 0.25

                row_idx += 1

        return out

    @staticmethod
    def _get_output_shape(num_intervals, width):
        NUM_SEQ_CHARS = 4
        return (num_intervals, width, NUM_SEQ_CHARS)

class LaceyFastaExtractorCA(BaseExtractor):
    def __init__(self, datafile, **kwargs):
        from pysam import FastaFile
        super(LaceyFastaExtractorCA, self).__init__(datafile, **kwargs)
        self.fasta = FastaFile(self._datafile)

    def _extract(self, intervals, out, **kwargs):
        for index, interval in enumerate(intervals):
            seq = self.fasta.fetch(str(interval.chrom), interval.start,
                                       interval.stop)
            row_idx = 0
            for base in seq:
                if base == 'A' or base == 'a':
                    out[index,row_idx, 0] = 1
                if base == 'C' or base == 'c':
                    out[index,row_idx, 1] = 1
                if base == 'G' or base == 'g':
                    out[index,row_idx, 1] = 1
                if base == 'T' or base == 't':
                    out[index,row_idx, 0] = 1
                if base == 'N' or base == 'n':
                    out[index,row_idx, 0] = 0.25
                    out[index,row_idx, 1] = 0.25
                    out[index,row_idx, 2] = 0.25
                    out[index,row_idx, 3] = 0.25

                row_idx += 1

        return out

    @staticmethod
    def _get_output_shape(num_intervals, width):
        NUM_SEQ_CHARS = 4
        return (num_intervals, width, NUM_SEQ_CHARS)

class LaceyFastaExtractorCADouble4Channel(BaseExtractor):
    def __init__(self, datafile, **kwargs):
        from pysam import FastaFile
        super(LaceyFastaExtractorCADouble4Channel, self).__init__(datafile, **kwargs)
        self.fasta = FastaFile(self._datafile)

    def _extract(self, intervals, out, **kwargs):
        for index, interval in enumerate(intervals):
            seq = self.fasta.fetch(str(interval.chrom), interval.start,
                                       interval.stop)
            row_idx = 0
            for base in seq:
                if base == 'A' or base == 'a':
                    out[index,row_idx, 0] = 0.5
                    out[index, row_idx, 3] = 0.5
                if base == 'C' or base == 'c':
                    out[index,row_idx, 1] = 0.5
                    out[index, row_idx, 2] = 0.5
                if base == 'G' or base == 'g':
                    out[index,row_idx, 1] = 0.5
                    out[index, row_idx, 2] = 0.5
                if base == 'T' or base == 't':
                    out[index,row_idx, 0] = 0.5
                    out[index, row_idx, 3] = 0.5
                if base == 'N' or base == 'n':
                    out[index,row_idx, 0] = 0.25
                    out[index,row_idx, 1] = 0.25
                    out[index,row_idx, 2] = 0.25
                    out[index,row_idx, 3] = 0.25

                row_idx += 1

        return out

    @staticmethod
    def _get_output_shape(num_intervals, width):
        NUM_SEQ_CHARS = 4
        return (num_intervals, width, NUM_SEQ_CHARS)

class LaceyFastaExtractorCA2Channel(BaseExtractor):
    def __init__(self, datafile, **kwargs):
        from pysam import FastaFile
        super(LaceyFastaExtractorCA2Channel, self).__init__(datafile, **kwargs)
        self.fasta = FastaFile(self._datafile)

    def _extract(self, intervals, out, **kwargs):
        for index, interval in enumerate(intervals):
            seq = self.fasta.fetch(str(interval.chrom), interval.start,
                                       interval.stop)
            row_idx = 0
            for base in seq:
                if base == 'A' or base == 'a':
                    out[index,row_idx, 0] = 1
                if base == 'C' or base == 'c':
                    out[index,row_idx, 1] = 1
                if base == 'G' or base == 'g':
                    out[index,row_idx, 1] = 1
                if base == 'T' or base == 't':
                    out[index,row_idx, 0] = 1
                if base == 'N' or base == 'n':
                    out[index,row_idx, 0] = 0.5
                    out[index,row_idx, 1] = 0.5

                row_idx += 1

        return out

    @staticmethod
    def _get_output_shape(num_intervals, width):
        NUM_SEQ_CHARS = 2
        return (num_intervals, width, NUM_SEQ_CHARS)

class LaceyFastaExtractorContentSkew(BaseExtractor):
    def __init__(self, datafile, **kwargs):
        from pysam import FastaFile
        super(LaceyFastaExtractorContentSkew, self).__init__(datafile, **kwargs)
        self.fasta = FastaFile(self._datafile)

    def _extract(self, intervals, out, **kwargs):
        for index, interval in enumerate(intervals):
            seq = self.fasta.fetch(str(interval.chrom), interval.start,
                                       interval.stop)
            row_idx = 0
            for base in seq:
                if base == 'A' or base == 'a':
                    out[index,row_idx, 0] = 0.5
                    out[index, row_idx, 2] = 0.5
                if base == 'C' or base == 'c':
                    out[index,row_idx, 1] = 0.5
                    out[index, row_idx, 2] = 0.5
                if base == 'G' or base == 'g':
                    out[index,row_idx, 1] = 0.5
                    out[index, row_idx, 3] = 0.5
                if base == 'T' or base == 't':
                    out[index,row_idx, 0] = 0.5
                    out[index, row_idx, 3] = 0.5
                if base == 'N' or base == 'n':
                    out[index,row_idx, 0] = 0.25
                    out[index,row_idx, 1] = 0.25
                    out[index,row_idx, 2] = 0.25
                    out[index,row_idx, 3] = 0.25

                row_idx += 1

        return out

    @staticmethod
    def _get_output_shape(num_intervals, width):
        NUM_SEQ_CHARS = 4
        return (num_intervals, width, NUM_SEQ_CHARS)

class LaceyFastaExtractorSkew(BaseExtractor):
    def __init__(self, datafile, **kwargs):
        from pysam import FastaFile
        super(LaceyFastaExtractorSkew, self).__init__(datafile, **kwargs)
        self.fasta = FastaFile(self._datafile)

    def _extract(self, intervals, out, **kwargs):
        for index, interval in enumerate(intervals):
            seq = self.fasta.fetch(str(interval.chrom), interval.start,
                                       interval.stop)
            row_idx = 0
            for base in seq:
                if base == 'A' or base == 'a' or base == 'C' or base == 'c':
                    out[index,row_idx, 0] = 1
                if base == 'G' or base == 'g' or base == 'T' or base == 't':
                    out[index,row_idx, 1] = 1
                if base == 'N' or base == 'n':
                    out[index,row_idx, 0] = 0.5
                    out[index,row_idx, 1] = 0.5
                row_idx += 1
        return out

    @staticmethod
    def _get_output_shape(num_intervals, width):
        NUM_SEQ_CHARS = 2
        return (num_intervals, width, NUM_SEQ_CHARS)

class LaceyFastaExtractorBinary(BaseExtractor):
    def __init__(self, datafile, **kwargs):
        from pysam import FastaFile
        super(LaceyFastaExtractorBinary, self).__init__(datafile, **kwargs)
        self.fasta = FastaFile(self._datafile)

    def _extract(self, intervals, out, **kwargs):
        for index, interval in enumerate(intervals):
            seq = self.fasta.fetch(str(interval.chrom), interval.start,
                                       interval.stop)
            row_idx = 0
            for base in seq:
                if base != 'N' and base != 'n':
                    out[index,row_idx, 0] = 1
                else:
                    out[index, row_idx, 0] = 0
                row_idx += 1
        return out

    @staticmethod
    def _get_output_shape(num_intervals, width):
        NUM_SEQ_CHARS = 1
        return (num_intervals, width, NUM_SEQ_CHARS)

class TsvReader:
    def __init__(self, tsv_file,
                 num_chr=False,
                 label_dtype=None,
                 mask_ambigous=None,
                 # task_prefix='task/',
                 incl_chromosomes=None,
                 excl_chromosomes=None,
                 chromosome_lens=None,
                 resize_width=None
                 ):
        """Reads a tsv/BED file in the following format:
        chr  start  stop  [task1  task2 ... ]

        Args:
          tsv_file: a tsv file with or without the header (i.e. BED file)
          num_chr: if True, remove the 'chr' prefix if existing in the chromosome names
          label_dtype: data type of the labels
          mask_ambigous: if specified, rows where `<task>==mask_ambigous` will be omitted
          incl_chromosomes (list of str): list of chromosomes to keep.
              Intervals from other chromosomes are dropped.
          excl_chromosomes (list of str): list of chromosomes to exclude.
              Intervals from other chromosomes are dropped.
          chromosome_lens (dict of int): dictionary with chromosome lengths
          resize_width (int): desired interval width. The resize fixes the center
              of the interval.

        """
        self.tsv_file = tsv_file
        self.num_chr = num_chr
        self.label_dtype = label_dtype
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes
        self.chromosome_lens = chromosome_lens
        self.resize_width = resize_width

        columns = list(pd.read_csv(self.tsv_file, nrows=0, sep='\t').columns)

        if not columns[0].startswith("CHR") and not columns[0].startswith("#CHR"):
            # No classes were provided
            self.columns = None
            self.tasknames = None
            skiprows = None
        else:
            # There exists a header
            self.columns = columns
            self.tasknames = list(columns)[3:]
            skiprows = [0]
            # self.tasknames = [c.replace(task_prefix, "") for c in columns if task_prefix in c]

        df_peek = pd.read_csv(self.tsv_file,
                              header=None,
                              nrows=1,
                              skiprows=skiprows,
                              comment='#',
                              sep='\t')
        self.n_tasks = df_peek.shape[1] - 3
        assert self.n_tasks >= 0

        if self.tasknames is not None:
            # make sure the task-names match
            assert self.n_tasks == len(self.tasknames)

        if self.label_dtype is None:
            dtypes = {i: d for i, d in enumerate([str, int, int])}
        else:
            dtypes = {i: d for i, d in enumerate([str, int, int] + [self.label_dtype] * self.n_tasks)}

        self.df = pd.read_csv(self.tsv_file,
                              header=None,
                              comment='#',
                              skiprows=skiprows,
                              dtype=dtypes,
                              sep='\t')
        if self.num_chr and self.df.iloc[0][0].startswith("chr"):
            self.df[0] = self.df[0].str.replace("^chr", "")
        if not self.num_chr and not self.df.iloc[0][0].startswith("chr"):
            self.df[0] = "chr" + self.df[0]

        if mask_ambigous is not None and self.n_tasks > 0:
            # exclude regions where only ambigous labels are present
            self.df = self.df[~np.all(self.df.iloc[:, 3:] == mask_ambigous, axis=1)]

        # omit data outside chromosomes
        if incl_chromosomes is not None:
            self.df = self.df[self.df[0].isin(incl_chromosomes)]
        if excl_chromosomes is not None:
            self.df = self.df[~self.df[0].isin(excl_chromosomes)]

        # make the chromosome name a categorical variable
        self.df[0] = pd.Categorical(self.df[0])

        # Skip intervals outside of the genome
        if self.chromosome_lens is not None:
            n_int = len(self.df)
            center = (self.df[1] + self.df[2]) // 2
            valid_seqs = ((center > self.resize_width // 2 + 1) &
                          (center < self.df[0].map(chromosome_lens).astype(int) - self.resize_width // 2 - 1))
            self.df = self.df[valid_seqs]

            if len(self.df) != n_int:
                print(f"Skipped {n_int - len(self.df)} intervals"
                      " outside of the genome size")

    def __getitem__(self, idx):
        """Returns (pybedtools.Interval, labels)
        """
        from pybedtools import Interval
        # TODO - speedup using: iat[idx, .]
        # interval = Interval(self.dfm.iat[idx, 0],  # chrom
        #                     self.dfm.iat[idx, 1],  # start
        #                     self.dfm.iat[idx, 2])  # end
        # intervals = [interval]
        # task = self.dfm.iat[idx, 3]  # task
        row = self.df.iloc[idx]
        interval = Interval(row[0], row[1], row[2])

        if self.n_tasks == 0:
            labels = {}
        else:
            labels = row.iloc[3:].values.astype(self.label_dtype)
        return interval, labels

    def __len__(self):
        return len(self.df)

    def get_target_names(self):
        return self.tasknames

    def get_targets(self):
        return self.df.iloc[:, 3:].values.astype(self.label_dtype)

    def shuffle_inplace(self):
        """Shuffle the interval
        """
        self.df = self.df.sample(frac=1)

    @classmethod
    def concat(self, tsv_datasets):
        """Concatenate multiple objects
        """
        for ds in tsv_datasets:
            assert ds.get_target_names() == tsv_datasets[0].get_target_names()
        obj = deepcopy(tsv_datasets[0])
        # concatenate the data-frames
        obj.tsv = pd.concat([ds.df for ds in tsv_datasets])
        return obj

    def append(self, tsv_dataset):
        """Append another tsv dataset. This returns a new
        object leaving the self object intact.
        """
        return self.concat([self, tsv_dataset])

class GuideTsvReader:
    def __init__(self, tsv_file,
                 num_chr=False,
                 incl_chromosomes=None,
                 excl_chromosomes=None,
                 chromosome_lens=None,
                 resize_width=None,
                 ):
        """Reads a tsv/BED file in the following format:
        chr  start  stop  [task1  task2 ... ]

        Args:
          tsv_file: a tsv file with or without the header (i.e. BED file)
          num_chr: if True, remove the 'chr' prefix if existing in the chromosome names
          label_dtype: data type of the labels
          mask_ambigous: if specified, rows where `<task>==mask_ambigous` will be omitted
          incl_chromosomes (list of str): list of chromosomes to keep.
              Intervals from other chromosomes are dropped.
          excl_chromosomes (list of str): list of chromosomes to exclude.
              Intervals from other chromosomes are dropped.
          chromosome_lens (dict of int): dictionary with chromosome lengths
          resize_width (int): desired interval width. The resize fixes the center
              of the interval.

        """
        self.tsv_file = tsv_file
        self.num_chr = num_chr
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes
        self.chromosome_lens = chromosome_lens
        self.resize_width = resize_width
        self.n_tasks = 1

        self.columns = None
        self.tasknames = None
        skiprows = None

        dtypes = {i: d for i, d in enumerate([str, int, int, str, float, str])}

        self.df = pd.read_csv(self.tsv_file,
                              header=None,
                              comment='#',
                              skiprows=skiprows,
                              dtype=dtypes,
                              sep='\t')
        if self.num_chr and self.df.iloc[0][0].startswith("chr"):
            self.df[0] = self.df[0].str.replace("^chr", "")
        if not self.num_chr and not self.df.iloc[0][0].startswith("chr"):
            self.df[0] = "chr" + self.df[0]

        # omit data outside chromosomes
        if incl_chromosomes is not None:
            self.df = self.df[self.df[0].isin(incl_chromosomes)]
        if excl_chromosomes is not None:
            self.df = self.df[~self.df[0].isin(excl_chromosomes)]

        # make the chromosome name a categorical variable
        self.df[0] = pd.Categorical(self.df[0])

        # Skip intervals outside of the genome
        if self.chromosome_lens is not None:
            n_int = len(self.df)
            center = (self.df[1] + self.df[2]) // 2
            valid_seqs = ((center > self.resize_width // 2 + 1) &
                          (center < self.df[0].map(chromosome_lens).astype(int) - self.resize_width // 2 - 1))
            self.df = self.df[valid_seqs]

            if len(self.df) != n_int:
                print(f"Skipped {n_int - len(self.df)} intervals"
                      " outside of the genome size")

    def __getitem__(self, idx):
        """Returns (pybedtools.Interval, labels)
        """
        from pybedtools import Interval
        # TODO - speedup using: iat[idx, .]
        # interval = Interval(self.dfm.iat[idx, 0],  # chrom
        #                     self.dfm.iat[idx, 1],  # start
        #                     self.dfm.iat[idx, 2])  # end
        # intervals = [interval]
        # task = self.dfm.iat[idx, 3]  # task
        row = self.df.iloc[idx]
        interval = Interval(row[0], row[1], row[2])

        labels = {}

        return interval, labels

    def __len__(self):
        return len(self.df)

    def get_target_names(self):
        return self.tasknames

    def get_targets(self):
        return self.df.iloc[:, 4].values.astype(float)

    def shuffle_inplace(self):
        """Shuffle the interval
        """
        self.df = self.df.sample(frac=1)

    @classmethod
    def concat(self, tsv_datasets):
        """Concatenate multiple objects
        """
        for ds in tsv_datasets:
            assert ds.get_target_names() == tsv_datasets[0].get_target_names()
        obj = deepcopy(tsv_datasets[0])
        # concatenate the data-frames
        obj.tsv = pd.concat([ds.df for ds in tsv_datasets])
        return obj

    def append(self, tsv_dataset):
        """Append another tsv dataset. This returns a new
        object leaving the self object intact.
        """
        return self.concat([self, tsv_dataset])



def _run_extractors(extractors, intervals, sum_tracks=False):
    """Helper function for StrandedProfile
    """
    if sum_tracks:
        out = sum([np.abs(ex(intervals, nan_as_zero=True))
                   for ex in extractors])[..., np.newaxis]  # keep the same dimension
    else:
        out = np.stack([np.abs(ex(intervals, nan_as_zero=True))
                        for ex in extractors], axis=-1)

    # Take the strand into account
    for i, interval in enumerate(intervals):
        if interval.strand == '-':
            out[i, :, :] = out[i, ::-1, ::-1]
    return out

@gin.configurable
class StrandedProfile(Dataset):
    # Added extra array that holds the additional bw channels
    # Specified in format [path_to_file1, path_to_file2, ...]
    def __init__(self, ds,
                 peak_width=200,
                 seq_width=None,
                 incl_chromosomes=None,
                 excl_chromosomes=None,
                 intervals_file=None,
                 intervals_format='bed',
                 include_metadata=True,
                 tasks=None,
                 include_classes=False,
                 shuffle=True,
                 interval_transformer=None,
                 track_transform=None,
                 total_count_transform=lambda x: np.log(1 + x),
                 exclude_dna=False,
                 added_tracks_to_exclude=None,
                 special_encode='none',# add option for special encoder
                 bias_fasta=False,
                 bias='ca2',
                 do_epi=False):
        """Dataset for loading the bigwigs and fastas
        Args:
          ds (bpnet.dataspecs.DataSpec): data specification containing the
            fasta file, bed files and bigWig file paths
          chromosomes (list of str): a list of chor
          peak_width: resize the bed file to a certain width
          intervals_file: if specified, use these regions to train the model.
            If not specified, the regions are inferred from the dataspec.
          intervals_format: interval_file format. Available: bed, bed3, bed3+labels
          shuffle: True
          track_transform: function to be applied to transform the tracks (shape=(batch, seqlen, channels))
          total_count_transform: transform to apply to the total counts
            TODO - shall we standardize this to have also the inverse operation?
        """
        if isinstance(ds, str):
            self.ds = DataSpec.load(ds)
        else:
            self.ds = ds
        self.peak_width = peak_width
        if seq_width is None:
            self.seq_width = peak_width
        else:
            self.seq_width = seq_width

        assert intervals_format in ['bed3', 'bed3+labels', 'bed']

        self.shuffle = shuffle
        self.intervals_file = intervals_file
        self.intervals_format = intervals_format
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes
        self.total_count_transform = total_count_transform
        self.track_transform = track_transform
        self.include_classes = include_classes
        self.exclude_dna = exclude_dna
        self.special_encode = special_encode
        self.bias_fasta = bias_fasta
        self.bias = bias
        self.do_epi = do_epi
        # not specified yet
        self.fasta_extractor = None
        self.bw_extractors = None
        self.bias_bw_extractors = None
        self.added_channel_extractors = None
        self.include_metadata = include_metadata
        self.interval_transformer = interval_transformer

        # Load chromosome lengths
        self.chrom_lens = _chrom_sizes(self.ds.fasta_file)

        if self.intervals_file is None:
            # concatenate the bed files
            self.dfm = pd.concat([TsvReader(task_spec.peaks,
                                            num_chr=False,
                                            incl_chromosomes=incl_chromosomes,
                                            excl_chromosomes=excl_chromosomes,
                                            chromosome_lens=self.chrom_lens,
                                            resize_width=max(self.peak_width, self.seq_width)
                                            ).df.iloc[:, :3].assign(task=task)
                                  for task, task_spec in self.ds.task_specs.items()
                                  if task_spec.peaks is not None])
            assert list(self.dfm.columns)[:4] == [0, 1, 2, "task"]
            if self.shuffle:
                self.dfm = self.dfm.sample(frac=1)
            self.tsv = None
            self.dfm_tasks = None
        else:
            self.tsv = TsvReader(self.intervals_file,
                                 num_chr=False,
                                 # optional
                                 label_dtype=int if self.intervals_format == 'bed3+labels' else None,
                                 mask_ambigous=-1 if self.intervals_format == 'bed3+labels' else None,
                                 # --------------------------------------------
                                 incl_chromosomes=incl_chromosomes,
                                 excl_chromosomes=excl_chromosomes,
                                 chromosome_lens=self.chrom_lens,
                                 resize_width=max(self.peak_width, self.seq_width)
                                 )
            if self.shuffle:
                self.tsv.shuffle_inplace()
            self.dfm = self.tsv.df  # use the data-frame from tsv
            self.dfm_tasks = self.tsv.get_target_names()

        # remember the tasks
        if tasks is None:
            self.tasks = list(self.ds.task_specs)
        else:
            self.tasks = tasks

        if self.include_classes:
            assert self.dfm_tasks is not None

        if self.dfm_tasks is not None:
            assert set(self.tasks).issubset(self.dfm_tasks)

        # setup bias maps per task
        self.task_bias_tracks = {task: [bias for bias, spec in self.ds.bias_specs.items()
                                        if task in spec.tasks]
                                 for task in self.tasks}

        # Setup added_tracks_to_exclude
        if not added_tracks_to_exclude:
            self.added_tracks_to_exclude = []
        else:
            self.added_tracks_to_exclude = added_tracks_to_exclude

    def __len__(self):
        return len(self.dfm)

    def get_targets(self):
        """
        'targets'
        """
        assert self.intervals_file is not None
        return self.tsv.get_targets()

    def __getitem__(self, idx):
        from pybedtools import Interval

        if self.fasta_extractor is None:
            # first call
            # Use normal fasta/bigwig extractors
            if self.special_encode=='at':
                self.fasta_extractor = LaceyFastaExtractor(self.ds.fasta_file)
            elif self.special_encode=='ag':
                self.fasta_extractor = LaceyFastaExtractorAG(self.ds.fasta_file)
            elif self.special_encode=='ca4':
                self.fasta_extractor = LaceyFastaExtractorCADouble4Channel(self.ds.fasta_file)
            elif self.special_encode=='ca2':
                self.fasta_extractor = LaceyFastaExtractorCA2Channel(self.ds.fasta_file)
            elif self.special_encode=='skew':
                self.fasta_extractor = LaceyFastaExtractorSkew(self.ds.fasta_file)
            elif self.special_encode=='binary':
                self.fasta_extractor = LaceyFastaExtractorBinary(self.ds.fasta_file)
            else:
                self.fasta_extractor = FastaExtractor(self.ds.fasta_file, use_strand=True)

            self.bw_extractors = {task: [BigwigExtractor(track) for track in task_spec.tracks]
                                  for task, task_spec in self.ds.task_specs.items() if task in self.tasks}

            self.bias_bw_extractors = {task: [BigwigExtractor(track) for track in task_spec.tracks]
                                       for task, task_spec in self.ds.bias_specs.items()}

            self.added_channel_extractors = [BigwigExtractor(track.path) for key, track in self.ds.added_track_dict.items() 
                                                                            if key not in self.added_tracks_to_exclude]

            if self.bias=='ca4':
                self.bias_fasta_extractor = LaceyFastaExtractorCADouble4Channel(self.ds.fasta_file)
            elif self.bias=='ca2':
                self.bias_fasta_extractor = LaceyFastaExtractorCA2Channel(self.ds.fasta_file)
            elif self.bias=='contentskew':
                self.bias_fasta_extractor = LaceyFastaExtractorContentSkew(self.ds.fasta_file)
            elif self.bias=='skew':
                self.bias_fasta_extractor = LaceyFastaExtractorSkew(self.ds.fasta_file)
            else:
                self.bias_fasta_extractor = LaceyFastaExtractorCA(self.ds.fasta_file)

        # Get the genomic interval for that particular datapoint
        interval = Interval(self.dfm.iat[idx, 0],  # chrom
                            self.dfm.iat[idx, 1],  # start
                            self.dfm.iat[idx, 2])  # end

        # Transform the input interval (for say augmentation...)
        if self.interval_transformer is not None:
            interval = self.interval_transformer(interval)

        # resize the intervals to the desired widths
        target_interval = resize_interval(deepcopy(interval), self.peak_width)
        seq_interval = resize_interval(deepcopy(interval), self.seq_width)

        # This only kicks in when we specify the taskname from dataspec
        # to the 3rd column. E.g. it doesn't apply when using intervals_file
        interval_from_task = self.dfm.iat[idx, 3] if self.intervals_file is None else ''

        # extract DNA sequence + one-hot encode it
        sequence = self.fasta_extractor([seq_interval])[0]
        
        ###############################
        # Extract the array holding extra values
        if self.added_channel_extractors and len(self.added_channel_extractors) > 0:
            added_rows = []
            for added_channel in self.added_channel_extractors:
                new_row = added_channel([seq_interval])[0]
                added_rows += [new_row]
            added_rows = np.swapaxes(np.array(added_rows), 0, 1)
            if self.exclude_dna is False:
                sequence = np.array([[*x, *y] for x,y in zip(sequence, added_rows)])
            else:
                sequence = np.array(added_rows)
        ###############################

        inputs = {"seq": sequence}

        # Sanity check
        if idx == 1: 
            print("Input Dimensions: ", inputs['seq'].shape)

        # exctract the profile counts from the bigwigs
        cuts = {f"{task}/profile": _run_extractors(self.bw_extractors[task],
                                                   [target_interval],
                                                   sum_tracks=spec.sum_tracks)[0]
                for task, spec in self.ds.task_specs.items() if task in self.tasks}
        if self.track_transform is not None:
            for task in self.tasks:
                cuts[f'{task}/profile'] = self.track_transform(cuts[f'{task}/profile'])

        # Add total number of counts
        for task in self.tasks:
            cuts[f'{task}/counts'] = self.total_count_transform(cuts[f'{task}/profile'].sum(0))

        if len(self.ds.bias_specs) > 0:
            # Extract the bias tracks
            biases = {bias_task: _run_extractors(self.bias_bw_extractors[bias_task],
                                                 [target_interval],
                                                 sum_tracks=spec.sum_tracks)[0]
                      for bias_task, spec in self.ds.bias_specs.items()}

            task_biases = {f"bias/{task}/profile": np.concatenate([biases[bt]
                                                                   for bt in self.task_bias_tracks[task]],
                                                                  axis=-1)
                           for task in self.tasks}

            if self.track_transform is not None:
                for task in self.tasks:
                    task_biases[f'bias/{task}/profile'] = self.track_transform(task_biases[f'bias/{task}/profile'])

            # Add total number of bias counts
            for task in self.tasks:
                task_biases[f'bias/{task}/counts'] = self.total_count_transform(task_biases[f'bias/{task}/profile'].sum(0))

            inputs = {**inputs, **task_biases}

        if self.bias_fasta:
            task_biases = {f'bias/{task}/bias_fasta': self.bias_fasta_extractor([seq_interval])[0]}
            inputs = {**inputs, **task_biases}

        if self.do_epi:
            abs_ds = DataSpec.load(self.ds.path)
            for track_name in abs_ds.epi_model_dirs.keys():
                epi_extractor = BigwigExtractor(
                    DataSpec.load(f'{abs_ds.epi_model_dirs[track_name].path}/dataspec.yml').task_specs[task].tracks[0])
                inputs[f'epi_tracks/{task}/{track_name}'] = epi_extractor([seq_interval])[0].reshape(-1,1)
                inputs[f'epi_models/{task}/{track_name}_model_dir'] = abs_ds.epi_model_dirs[track_name].path
            inputs[f'seq_model_dir/{task}'] = abs_ds.dna_model_dir

        if self.include_classes:
            # Optionally, add binary labels from the additional columns in the tsv intervals file
            classes = {f"{task}/class": self.dfm.iat[idx, i + 3]
                       for i, task in enumerate(self.dfm_tasks) if task in self.tasks}
            cuts = {**cuts, **classes}

        out = {"inputs": inputs,
               "targets": cuts}

        if self.include_metadata:
            # remember the metadata (what genomic interval was used)
            out['metadata'] = {"range": GenomicRanges(chr=target_interval.chrom,
                                                      start=target_interval.start,
                                                      end=target_interval.stop,
                                                      id=idx,
                                                      strand=(target_interval.strand
                                                              if target_interval.strand is not None
                                                              else "*"),
                                                      ),
                               "interval_from_task": interval_from_task}
        return out

@gin.configurable
class GuideActivityProfile(Dataset):
    # Added extra array that holds the additional bw channels
    # Specified in format [path_to_file1, path_to_file2, ...]
    def __init__(self, ds,
                 peak_width=200,
                 seq_width=None,
                 incl_chromosomes=None,
                 excl_chromosomes=None,
                 intervals_file=None,
                 intervals_format='bed',
                 include_metadata=True,
                 tasks=None,
                 include_classes=False,
                 shuffle=True,
                 interval_transformer=None,
                 track_transform=None,
                 #total_count_transform=lambda x: np.log(1 + x),
                 exclude_dna=False,
                 added_tracks_to_exclude=None):
        """Dataset for loading the bigwigs and fastas
        Args:
          ds (bpnet.dataspecs.DataSpec): data specification containing the
            fasta file, bed files and bigWig file paths
          chromosomes (list of str): a list of chor
          peak_width: resize the bed file to a certain width
          intervals_file: if specified, use these regions to train the model.
            If not specified, the regions are inferred from the dataspec.
          intervals_format: interval_file format. Available: bed, bed3, bed3+labels
          shuffle: True
          track_transform: function to be applied to transform the tracks (shape=(batch, seqlen, channels))
          total_count_transform: transform to apply to the total counts
            TODO - shall we standardize this to have also the inverse operation?
        """
        if isinstance(ds, str):
            self.ds = DataSpec.load(ds)
        else:
            self.ds = ds
        self.peak_width = peak_width
        if seq_width is None:
            self.seq_width = peak_width
        else:
            self.seq_width = seq_width

        assert intervals_format == 'bed'

        self.shuffle = shuffle
        self.intervals_file = intervals_file
        self.intervals_format = intervals_format
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes
        self.track_transform = track_transform
        self.include_classes = include_classes
        self.exclude_dna = exclude_dna
        # not specified yet
        self.fasta_extractor = None
        self.bw_extractors = None
        self.bias_bw_extractors = None
        self.added_channel_extractors = None
        self.include_metadata = include_metadata
        self.interval_transformer = interval_transformer

        # Load chromosome lengths
        self.chrom_lens = _chrom_sizes(self.ds.fasta_file)

        if self.intervals_file is None:
            self.tsv = GuideTsvReader(ds.task_specs["rep1"].peaks,
                                 incl_chromosomes=incl_chromosomes,
                                 excl_chromosomes=excl_chromosomes,
                                 chromosome_lens=self.chrom_lens,
                                 resize_width=max(self.peak_width, self.seq_width)
                                 )
        else:
            self.tsv = GuideTsvReader(self.intervals_file,
                                 incl_chromosomes=incl_chromosomes,
                                 excl_chromosomes=excl_chromosomes,
                                 chromosome_lens=self.chrom_lens,
                                 resize_width=max(self.peak_width, self.seq_width)
                                 )
        if self.shuffle:
            self.tsv.shuffle_inplace()
        self.dfm = self.tsv.df  # use the data-frame from tsv

        # remember the tasks
        if tasks is None:
            self.tasks = list(self.ds.task_specs)
        else:
            self.tasks = tasks

        # setup bias maps per task
        self.task_bias_tracks = {task: [bias for bias, spec in self.ds.bias_specs.items()
                                        if task in spec.tasks]
                                 for task in self.tasks}

        # Setup added_tracks_to_exclude
        if not added_tracks_to_exclude:
            self.added_tracks_to_exclude = []
        else:
            self.added_tracks_to_exclude = added_tracks_to_exclude

    def __len__(self):
        return len(self.dfm)

    def get_targets(self):
        """
        'targets'
        """
        assert self.intervals_file is not None
        return self.tsv.get_targets()

    def __getitem__(self, idx):
        from pybedtools import Interval

        if self.fasta_extractor is None:
            # first call
            # Use normal fasta/bigwig extractors
            self.fasta_extractor = FastaExtractor(self.ds.fasta_file, use_strand=True)

            self.bw_extractors = {task: [BigwigExtractor(track) for track in task_spec.tracks]
                                  for task, task_spec in self.ds.task_specs.items() if task in self.tasks}

            self.added_channel_extractors = [BigwigExtractor(track.path) for key, track in
                                             self.ds.added_track_dict.items()
                                             if key not in self.added_tracks_to_exclude]

        # Get the genomic interval for that particular datapoint
        interval = Interval(self.dfm.iat[idx, 0],  # chrom
                            self.dfm.iat[idx, 1],  # start
                            self.dfm.iat[idx, 2])  # end

        # Transform the input interval (for say augmentation...)
        if self.interval_transformer is not None:
            interval = self.interval_transformer(interval)

        # resize the intervals to the desired widths
        seq_interval = resize_interval(deepcopy(interval), self.seq_width)

        # extract DNA sequence + one-hot encode it
        sequence = self.fasta_extractor([seq_interval])[0]

        #extract guide track
        #guide guide location from col 3 of df
        guide_chrom, guide_start, guide_end, guide_strand = self.dfm.iat[idx, 3].split('_')

        seq_indexes = list(range(seq_interval.start, seq_interval.stop))
        guide_array = pd.DataFrame()
        guide_array[0] = seq_indexes
        guide_array[1] = 0
        guide_array[2] = 0
        guide_array = guide_array.set_index(0)
        if guide_strand == '+':
            guide_array[1].loc[int(guide_start):int(guide_end)] = 1
        else:
            guide_array[2].loc[int(guide_start):int(guide_end)] = 1
        #guide_array = guide_array[[1,2]].to_numpy()
        #added_rows = [guide_array[[1]].to_numpy(),guide_array[[2]].to_numpy()]

        ###############################
        # Extract the array holding extra values
        if self.added_channel_extractors and len(self.added_channel_extractors) > 0:
            added_rows = []
            for added_channel in self.added_channel_extractors:
                new_row = added_channel([seq_interval])[0]
                added_rows += [new_row]
            added_rows = np.swapaxes(np.array(added_rows), 0, 1)



            if self.exclude_dna is False:
                #sequence = np.array([[*x, *y] for x, y in zip(sequence, added_rows)])
                sequence = np.array([[*w, *x, *y, *z] for w, x, y, z in zip(sequence, guide_array[[1]].to_numpy(), guide_array[[2]].to_numpy(), added_rows)])
            else:
                #sequence = np.array(added_rows)
                sequence = np.array([[*w, *x, *y] for w, x, y in zip(guide_array[[1]].to_numpy(), guide_array[[2]].to_numpy(), added_rows)])
        ###############################

        inputs = {"seq": sequence}

        # Sanity check
        if idx == 1:
            print("Input Dimensions: ", inputs['seq'].shape)

        # exctract the activity score

        cuts = {f"{self.tasks[0]}/activity": np.array([self.dfm.iat[idx, 4]])}

        out = {"inputs": inputs,
               "targets": cuts}

        if self.include_metadata:
            # remember the metadata (what genomic interval was used)
            guide_interval = Interval(guide_chrom,
                                      int(guide_start),
                                      int(guide_end),
                                      strand = guide_strand)
            out['metadata'] = {"range": GenomicRanges(chr=seq_interval.chrom,
                                                      start=seq_interval.start,
                                                      end=seq_interval.stop,
                                                      id=idx,
                                                      strand=(seq_interval.strand
                                                              if seq_interval.strand is not None
                                                              else "*"),
                                                      ),
                               "guide_range": GenomicRanges(chr=guide_interval.chrom,
                                                           start=guide_interval.start,
                                                           end=guide_interval.stop,
                                                           id=self.dfm.iat[idx, 3],
                                                           strand=(guide_interval.strand
                                                                   if guide_interval.strand is not None
                                                                   else "*"),
                                                           ),
                               "interval_from_task": ''}

        return out

# -------------------------------------------------------
# final datasets returning a (train, validation) tuple
@gin.configurable
def bpnet_data(dataspec,
               peak_width=1000,
               intervals_file=None,
               intervals_format='bed',
               seq_width=None,
               shuffle=True,
               total_count_transform=lambda x: np.log(1 + x),
               track_transform=None,
               include_metadata=False,
               valid_chr=['chr2', 'chr3', 'chr4'],
               test_chr=['chr1', 'chr8', 'chr9'],
               exclude_chr=[],
               augment_interval=True,
               interval_augmentation_shift=200,
               tasks=None,
               guide_data=False):
    """BPNet default data-loader

    Args:
      tasks: specify a subset of the tasks to use in the dataspec.yml. If None, all tasks will be specified.
    """
    from bpnet.metrics import BPNetMetric, PeakPredictionProfileMetric, pearson_spearman
    # test and valid shouldn't be in the valid or test sets
    for vc in valid_chr:
        assert vc not in exclude_chr
    for vc in test_chr:
        assert vc not in exclude_chr

    dataspec = DataSpec.load(dataspec)

    if tasks is None:
        tasks = list(dataspec.task_specs)

    if augment_interval:
        interval_transformer = IntervalAugmentor(max_shift=interval_augmentation_shift,
                                                 flip_strand=True)
    else:
        interval_transformer = None

    # get the list of all chromosomes from the fasta file
    all_chr = _chrom_names(dataspec.fasta_file)


    if not guide_data:
        return (StrandedProfile(dataspec, peak_width,
                                intervals_file=intervals_file,
                                intervals_format=intervals_format,
                                seq_width=seq_width,
                                include_metadata=include_metadata,
                                incl_chromosomes=[c for c in all_chr
                                                  if c not in valid_chr + test_chr + exclude_chr],
                                excl_chromosomes=valid_chr + test_chr + exclude_chr,
                                tasks=tasks,
                                shuffle=shuffle,
                                track_transform=track_transform,
                                total_count_transform=total_count_transform,
                                interval_transformer=interval_transformer),
                [('valid-peaks', StrandedProfile(dataspec,
                                                 peak_width,
                                                 intervals_file=intervals_file,
                                                 intervals_format=intervals_format,
                                                 seq_width=seq_width,
                                                 include_metadata=include_metadata,
                                                 incl_chromosomes=valid_chr,
                                                 tasks=tasks,
                                                 interval_transformer=interval_transformer,
                                                 shuffle=shuffle,
                                                 track_transform=track_transform,
                                                 total_count_transform=total_count_transform)),
                 ('train-peaks', StrandedProfile(dataspec, peak_width,
                                                 intervals_file=intervals_file,
                                                 intervals_format=intervals_format,
                                                 seq_width=seq_width,
                                                 include_metadata=include_metadata,
                                                 incl_chromosomes=[c for c in all_chr
                                                                   if c not in valid_chr + test_chr + exclude_chr],
                                                 excl_chromosomes=valid_chr + test_chr + exclude_chr,
                                                 tasks=tasks,
                                                 interval_transformer=interval_transformer,
                                                 shuffle=shuffle,
                                                 track_transform=track_transform,
                                                 total_count_transform=total_count_transform)),
                 ])
    else:
        return (GuideActivityProfile(dataspec, peak_width,
                                intervals_file=dataspec.task_specs["rep1"].peaks,
                                seq_width=seq_width,
                                include_metadata=include_metadata,
                                incl_chromosomes=[c for c in all_chr
                                                  if c not in valid_chr + test_chr + exclude_chr],
                                excl_chromosomes=valid_chr + test_chr + exclude_chr,
                                tasks=tasks,
                                shuffle=shuffle,
                                interval_transformer=interval_transformer),
                [('valid-peaks', GuideActivityProfile(dataspec,
                                                 peak_width,
                                                 intervals_file=dataspec.task_specs["rep1"].peaks,
                                                 intervals_format=intervals_format,
                                                 seq_width=seq_width,
                                                 include_metadata=include_metadata,
                                                 incl_chromosomes=valid_chr,
                                                 tasks=tasks,
                                                 interval_transformer=interval_transformer,
                                                 shuffle=shuffle,
                                                 track_transform=track_transform)),
                 ('train-peaks', GuideActivityProfile(dataspec, peak_width,
                                                 intervals_file=dataspec.task_specs["rep1"].peaks,
                                                 intervals_format=intervals_format,
                                                 seq_width=seq_width,
                                                 include_metadata=include_metadata,
                                                 incl_chromosomes=[c for c in all_chr
                                                                   if c not in valid_chr + test_chr + exclude_chr],
                                                 excl_chromosomes=valid_chr + test_chr + exclude_chr,
                                                 tasks=tasks,
                                                 interval_transformer=interval_transformer,
                                                 shuffle=shuffle,
                                                 track_transform=track_transform)),
                 ])

@gin.configurable
def bpnet_data_gw(dataspec,
                  intervals_file=None,
                  peak_width=200,
                  seq_width=None,
                  shuffle=True,
                  track_transform=None,
                  total_count_transform=lambda x: np.log(1 + x),
                  include_metadata=False,
                  include_classes=False,
                  tasks=None,
                  valid_chr=['chr2', 'chr3', 'chr4'],
                  test_chr=['chr1', 'chr8', 'chr9'],
                  exclude_chr=[]):
    """Genome-wide bpnet data
    """
    # NOTE = only chromosomes from chr1-22 and chrX and chrY are considered here
    # (e.g. all other chromosomes like ChrUn... are omitted)
    from bpnet.metrics import BPNetMetric, PeakPredictionProfileMetric, pearson_spearman
    # test and valid shouldn't be in the valid or test sets
    for vc in valid_chr:
        assert vc not in exclude_chr
    for vc in test_chr:
        assert vc not in exclude_chr

    dataspec = DataSpec.load(dataspec)

    # get the list of all chromosomes from the fasta file
    all_chr = _chrom_names(dataspec.fasta_file)

    if tasks is None:
        tasks = list(dataspec.task_specs)

    train = StrandedProfile(dataspec, peak_width,
                            seq_width=seq_width,
                            intervals_file=intervals_file,
                            intervals_format='bed3+labels',
                            include_metadata=include_metadata,
                            include_classes=include_classes,
                            tasks=tasks,
                            incl_chromosomes=[c for c in all_chr
                                              if c not in valid_chr + test_chr + exclude_chr],
                            excl_chromosomes=valid_chr + test_chr + exclude_chr,
                            shuffle=shuffle,
                            track_transform=track_transform,
                            total_count_transform=total_count_transform)

    valid = [('train-valid-genome-wide',
              StrandedProfile(dataspec, peak_width,
                              seq_width=seq_width,
                              intervals_file=intervals_file,
                              intervals_format='bed3+labels',
                              include_metadata=include_metadata,
                              include_classes=include_classes,
                              tasks=tasks,
                              incl_chromosomes=valid_chr,
                              shuffle=shuffle,
                              track_transform=track_transform,
                              total_count_transform=total_count_transform))]
    if include_classes:
        # Only use binary classification for genome-wide evaluation
        valid = valid + [('valid-genome-wide',
                          StrandedProfile(dataspec, peak_width,
                                          seq_width=seq_width,
                                          intervals_file=intervals_file,
                                          intervals_format='bed3+labels',
                                          include_metadata=include_metadata,
                                          include_classes=True,
                                          tasks=tasks,
                                          incl_chromosomes=valid_chr,
                                          shuffle=shuffle,
                                          track_transform=track_transform,
                                          total_count_transform=total_count_transform))]

    # Add also the peak regions
    valid = valid + [
        ('valid-peaks', StrandedProfile(dataspec, peak_width,
                                        seq_width=seq_width,
                                        intervals_file=None,
                                        intervals_format='bed3+labels',
                                        include_metadata=include_metadata,
                                        tasks=tasks,
                                        include_classes=False,  # dataspec doesn't contain labels
                                        incl_chromosomes=valid_chr,
                                        shuffle=shuffle,
                                        track_transform=track_transform,
                                        total_count_transform=total_count_transform)),
        ('train-peaks', StrandedProfile(dataspec, peak_width,
                                        seq_width=seq_width,
                                        intervals_file=None,
                                        intervals_format='bed3+labels',
                                        include_metadata=include_metadata,
                                        tasks=tasks,
                                        include_classes=False,  # dataspec doesn't contain labels
                                        incl_chromosomes=[c for c in all_chr
                                                          if c not in valid_chr + test_chr + exclude_chr],
                                        excl_chromosomes=valid_chr + test_chr + exclude_chr,
                                        shuffle=shuffle,
                                        track_transform=track_transform,
                                        total_count_transform=total_count_transform)),
        # use the default metric for the peak sets
    ]
    return train, valid

@gin.configurable
class SeqClassification(Dataset):
    """
    Args:
        intervals_file: bed3+<columns> file containing intervals+labels
        fasta_file: file path; Genome sequence
        label_dtype: label data type
        num_chr_fasta: if True, the tsv-loader will make sure that the chromosomes
          don't start with chr
    """

    def __init__(self, intervals_file,
                 fasta_file,
                 incl_chromosomes=None,
                 excl_chromosomes=None,
                 auto_resize_len=None,
                 num_chr_fasta=False):
        self.num_chr_fasta = num_chr_fasta
        self.intervals_file = intervals_file
        self.fasta_file = fasta_file
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes
        self.auto_resize_len = auto_resize_len

        self.tsv = TsvReader(self.intervals_file,
                             num_chr=self.num_chr_fasta,
                             label_dtype=int,
                             mask_ambigous=-1,
                             incl_chromosomes=incl_chromosomes,
                             excl_chromosomes=excl_chromosomes,
                             )
        self.fasta_extractor = None

    def __len__(self):
        return len(self.tsv)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            self.fasta_extractor = FastaExtractor(self.fasta_file)

        interval, labels = self.tsv[idx]

        if self.auto_resize_len:
            # automatically resize the sequence to cerat
            interval = resize_interval(interval, self.auto_resize_len)

        # Run the fasta extractor
        seq = np.squeeze(self.fasta_extractor([interval]))

        return {
            "inputs": {"seq": seq},
            "targets": labels,
            "metadata": {
                "range": GenomicRanges(chr=interval.chrom,
                                        start=interval.start,
                                        end=interval.stop,
                                        id=str(idx),
                                        strand=(interval.strand
                                                if interval.strand is not None
                                                else "*"),
                                        ),
                "interval_from_task": ''
            }
        }

    def get_targets(self):
        return self.tsv.get_targets()


@gin.configurable
def chrom_dataset(dataset_cls,
                  valid_chr=['chr2', 'chr3', 'chr4'],
                  holdout_chr=['chr1', 'chr8', 'chr9']):
    return (dataset_cls(excl_chromosomes=valid_chr + holdout_chr),
            dataset_cls(incl_chromosomes=valid_chr))


def get(name):
    return get_from_module(name, globals())

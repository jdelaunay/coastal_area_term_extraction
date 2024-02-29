# A Coastal area Term Extraction corpus

[Description](#Description) | [Data structure](#Data_structure) | [Annotations](#Annotations) | [Additional information](#Additional_information) | [License](#License) | [Contributors](#contributors)

## <a name="Description"></a> Description

We present a manually annotated dataset for Automatic Term Extraction (ATE) from scientific abstracts pertaining to coastal areas. This corpus comprises 195 abstracts preannotated using three Knowledge Bases (KBs): [AGROVOC](https://agrovoc.fao.org/browse/agrovoc/en/), [GEMET](https://www.eionet.europa.eu/gemet/en/about/), and [TAXREF-RD](https://inpn.mnhn.fr/programme/referentiel-taxonomique-taxref?lg=en), and further revised by a human annotator. We only annotated sentences pertaining to the functioning of littoral systems. Out of the 1,960 sentences, 1,149 contain annotated terms. All abstracts are in English. We conduct experiments using state-of-the-art (SOTA) models for ATE.

## <a name="Data_structure"></a> Data structure

The IOB annotations for the dataset follow the same format as ACTER. Additionally, a list of all unique terms is provided. Annotations are available for entire abstracts, individual sentences, and only sentences containing annotated terms.

```
.
├── README.md
└── data
    ├── annotations
    │   ├── sequential_annotations
    │   │   ├── iob_annotations
    │   │   │   ├── 1985_155.tsv
    │   │   │   ├── 1985_336.tsv
    │   │   │   └── ...
    │   │   └── iob_annotations_sents_wo_empty
    │   │       ├── 1985_155_1.tsv
    │   │       ├── 1985_155_2.tsv
    │   │       └── ...
    │   └── unique_annotations_lists
    │       └── en_terms.tsv
    ├── sents_tokenized
    │   ├── 1985_155_0.txt
    │   ├── 1985_155_1.txt
    │   └── ...
    ├── sents_tokenized_wo_empty
    │   ├── 1985_155_1.txt
    │   ├── 1985_155_2.txt
    │   └── ...
    └── texts_tokenized
        ├── 1985_155.txt
        ├── 1985_336.txt
        └── ...

11 directories, 6610 files
```

## <a name="Annotations"></a> Annotations

We collected 60,000+ abstracts from Scopus, from 1980 to 2023, containing the terms "coastal area" or "littoral". We randomly selected 195 among them, and used the annotator tool from [Agroportal](https://agroportal.lirmm.fr/) to preannotate terms appearing in three KBs: AGROVOC, a thesaurus on agronomy, GEMET, a general  environmental thesaurus,  and TAXREF-RD, a French national taxonomical register for fauna, flora and fungus, that covers mainland France and overseas territories.
We then manually annotated these abstracts using these pre-annotations, with the [INCEpTION](https://inception-project.github.io/) annotation tool. We focused only on sentences that informed us on the functioning on the coastal areas, meaning we did not annotate most of the parts that described methods.

## <a name="Additional_information"></a> Additional information

```
<to be added>
```

## <a name="License"></a> License

* *License*: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
* *Attribution*: Please cite the following paper if you use this dataset in your research:

```
<to be added>
```

## <a name="Contributors"></a> Contributors

- [DELAUNAY Julien](https://github.com/jdelaunay)
- [TRAN Thi Hong Hanh](https://github.com/honghanhh)
- [GONZÁLEZ-GALLARDO Carlos-Emiliano](https://github.com/cic4k)
- [DUCOS Mathilde](https://github.com/mducos)
- Prof. [Nicolas SIDERE](https://github.com/nsidere)
- Prof. [Antoine DOUCET](https://github.com/antoinedoucet)
- Prof. Olivier DE VIRON

# A Coastal area Term Extraction corpus
## Description
We present a manually annotated dataset for Automatic Term Extraction (ATE) from scientific abstracts pertaining to coastal areas. This corpus comprises 195 abstracts preannotated using three Knowledge Bases (KBs): AGROVOC, GEMET, and TAXREF-RD, and further revised by a human annotator. We only annotated sentences pertaining to the functioning of littoral systems. Out of the 1,960 sentences, 1,149 contain annotated terms. All abstracts are in English. We conduct experiments using state-of-the-art (SOTA) models for ATE.

## Data structure
The IOB annotations for the dataset follow the same format as ACTER. Additionally, a list of all unique terms is provided. Annotations are available for entire abstracts, individual sentences, and only sentences containing annotated terms.

## Annotations
We collected 60,000+ abstracts from Scopus, from 1980 to 2023, containing the terms "coastal area" or "littoral". We randomly selected 195 among them, and used the annotator tool from [Agroportal](https://agroportal.lirmm.fr/) to preannotate terms appearing in three KBs: AGROVOC, a thesaurus on agronomy, GEMET, a general  environmental thesaurus,  and TAXREF-RD, a French national taxonomical register for fauna, flora and fungus, that covers mainland France and overseas territories.
We then manually annotated these abstracts using these pre-annotations, with the [INCEpTION](https://inception-project.github.io/) annotation tool. We focused only on sentences that informed us on the functioning on the coastal areas, meaning we did not annotate most of the parts that described methods.

## Additional information

## License
* *License*: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) 
                (https://creativecommons.org/licenses/by-nc-sa/4.0/)
                
## Contributors
- [DELAUNAY Julien](https://github.com/jdelaunay)
- [TRAN Thi Hong Hanh](https://github.com/honghanhh)
- [GONZÁLEZ-GALLARDO Carlos-Emiliano](https://github.com/cic4k)
- [DUCOS Mathilde](https://github.com/mducos)
- Prof. [Nicolas SIDERE](https://github.com/nsidere)
- Prof. [Antoine DOUCET](https://github.com/antoinedoucet)
- Prof. Olivier DE VIRON
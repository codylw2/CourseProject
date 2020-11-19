# IR Competition

The competition is based on the CORD-19 dataset for supporting research on COVID-19. You're free to use both supervised or unsupervised approaches for this competition.

The test files can be obtained from https://drive.google.com/file/d/1FCW8fmcneow5yyDgApkPIGM-r2x6OFkm/view?usp=sharing

The train files can be obtained from https://drive.google.com/file/d/1E_Y-MkNvoOYoCZUZZa8JJ3ExiHYTfKTo/view?usp=sharing

Dataset format:

The following files are available in both the `train` and `test` folders:

- `documents/embeddings.zip`:  A collection of precomputed [SPECTER](https://arxiv.org/abs/2004.07180) document embeddings for each CORD-19 paper
- `documents/document_parses.tar.gz`: A collection of JSON files that contain full text parses of a subset of CORD-19 papers
- `documents/metadata.csv`: Metadata for all CORD-19 papers.
- `queries.xml`: Queries. Each 'topic' is a single query. You may use any (or a combination) of the fields (query, question, narrative) for each query.

The train folder also contains a relevance judgements file. Each row has a `topic/query_id doc_uid relevance_score`. 

When `embeddings.tar.gz` is uncompressed, it is a 769-column CSV file, where the first column is the `uid` and the remaining columns correspond to a 768-dimensional document embedding.  For example:

```
ug7v899j,-2.939983606338501,-6.312200546264648,-1.0459030866622925,5.164162635803223,-0.32564637064933777,-2.507413387298584,1.735608696937561,1.9363566637039185,0.622501015663147,1.5613162517547607,...
```



When `document_parses.tar.gz` is uncompressed, it is a directory:

```
|-- document_parses/
    |-- pdf_json/
        |-- 80013c44d7d2d3949096511ad6fa424a2c740813.json
        |-- bfe20b3580e7c539c16ce4b1e424caf917d3be39.json
        |-- ...
    |-- pmc_json/
        |-- PMC7096781.xml.json
        |-- PMC7118448.xml.json
        |-- ...
```



### Example usage of documents

It is recommended to primarily use `metadata.csv` & augment data when needed with full text in `document_parses/`.  For example, let's say we wanted to collect a bunch of Titles, Abstracts, and Introductions of papers.  In Python, such a script might look like:

```
import csv
import os
import json
from collections import defaultdict

uid_to_text = defaultdict(list)

# open the file
with open('metadata.csv') as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:
    
        # access some metadata
        uid = row['uid']
        title = row['title']
        abstract = row['abstract']
        authors = row['authors'].split('; ')

        # access the full text (if available) for Intro
        introduction = []
        if row['pdf_json_files']:
            for json_path in row['pdf_json_files'].split('; '):
                with open(json_path) as f_json:
                    full_text_dict = json.load(f_json)
                    
                    # grab introduction section from *some* version of the full text
                    for paragraph_dict in full_text_dict['body_text']:
                        paragraph_text = paragraph_dict['text']
                        section_name = paragraph_dict['section']
                        if 'intro' in section_name.lower():
                            introduction.append(paragraph_text)

                    # stop searching other copies of full text if already got introduction
                    if introduction:
                        break

        # save for later usage
        uid_to_text[uid].append({
            'title': title,
            'abstract': abstract,
            'introduction': introduction
        })
```

### `metadata.csv` overview

It is recommended to use `metadata.csv` as the starting point.  This file is comma-separated with the following columns:

- `uid`:  A `str`-valued field that assigns a unique identifier to each CORD-19 paper.  This is not necessariy unique per row, which is explained in the FAQs.
- `sha`:  A `List[str]`-valued field that is the SHA1 of all PDFs associated with the CORD-19 paper.  Most papers will have either zero or one value here (since we either have a PDF or we don't), but some papers will have multiple.  For example, the main paper might have supplemental information saved in a separate PDF.  Or we might have two separate PDF copies of the same paper.  If multiple PDFs exist, their SHA1 will be semicolon-separated (e.g. `'4eb6e165ee705e2ae2a24ed2d4e67da42831ff4a; d4f0247db5e916c20eae3f6d772e8572eb828236'`)
- `source_x`:  A `List[str]`-valued field that is the names of sources from which we received this paper.  Also semicolon-separated.  For example, `'ArXiv; Elsevier; PMC; WHO'`.  There should always be at least one source listed.
- `title`:  A `str`-valued field for the paper title
- `doi`: A `str`-valued field for the paper DOI
- `pmcid`: A `str`-valued field for the paper's ID on PubMed Central.  Should begin with `PMC` followed by an integer.
- `pubmed_id`: An `int`-valued field for the paper's ID on PubMed.  
- `license`: A `str`-valued field with the most permissive license we've found associated with this paper.  Possible values include:  `'cc0', 'hybrid-oa', 'els-covid', 'no-cc', 'cc-by-nc-sa', 'cc-by', 'gold-oa', 'biorxiv', 'green-oa', 'bronze-oa', 'cc-by-nc', 'medrxiv', 'cc-by-nd', 'arxiv', 'unk', 'cc-by-sa', 'cc-by-nc-nd'`
- `abstract`: A `str`-valued field for the paper's abstract
- `publish_time`:  A `str`-valued field for the published date of the paper.  This is in `yyyy-mm-dd` format.  Not always accurate as some publishers will denote unknown dates with future dates like `yyyy-12-31`
- `authors`:  A `List[str]`-valued field for the authors of the paper.  Each author name is in `Last, First Middle` format and semicolon-separated.
- `journal`:  A `str`-valued field for the paper journal.  Strings are not normalized (e.g. `BMJ` and `British Medical Journal` can both exist). Empty string if unknown.
- `mag_id`:  Deprecated, but originally an `int`-valued field for the paper as represented in the Microsoft Academic Graph.
- `who_covidence_id`:  A `str`-valued field for the ID assigned by the WHO for this paper.  Format looks like `#72306`. 
- `arxiv_id`:  A `str`-valued field for the arXiv ID of this paper.
- `pdf_json_files`:  A `List[str]`-valued field containing paths from the root of the current data dump version to the parses of the paper PDFs into JSON format.  Multiple paths are semicolon-separated.  Example: `document_parses/pdf_json/4eb6e165ee705e2ae2a24ed2d4e67da42831ff4a.json; document_parses/pdf_json/d4f0247db5e916c20eae3f6d772e8572eb828236.json`
- `pmc_json_files`:  A `List[str]`-valued field.  Same as above, but corresponding to the full text XML files downloaded from PMC, parsed into the same JSON format as above.
- `url`: A `List[str]`-valued field containing all URLs associated with this paper.  Semicolon-separated.
- `s2_id`:  A `str`-valued field containing the Semantic Scholar ID for this paper.  Can be used with the Semantic Scholar API (e.g. `s2_id=9445722` corresponds to `http://api.semanticscholar.org/corpusid:9445722`)


### Questions about CORD-19

#### Why can the same `uid` appear in multiple rows?
They are the same paper, but sent from different sources. Different rows might have different information about the paper. All of this data is representative of the same paper. 

#### What should we do if both PDF and PMC JSONs exist?  Or if there are multiple PDF JSONs?
These are different attempts/views to represent the same paper/document.  Some are going to be higher quality than others.  Treat these are separate representations of the same document – you can choose to use one, both, neither (i.e. just use the metadata fields).  On average, the PMC JSONs are cleaner than the PDF JSONs but that’s not necessarily true. 



***Submission Instructions*** : Follow the same instructions as for the MPs -- create a private copy of this repo and add a webhook to connect to LiveDataLab. Please add a  file named `predictions.txt` containing the predicted relevance scores on the test queries. Please submit The file should be in the same format as train/qrels.txt, i.e `topic/query_id doc_uid relevance_score`. You should submit the scores for the top 1000 documents per query.

References

```
 @inproceedings{wang-lo-2020-cord19,
    title={{CORD-19: The Covid-19 Open Research Dataset}},
    author={Lucy Lu Wang and Kyle Lo and Yoganand Chandrasekhar and Russell Reas and Jiangjiang Yang and Darrin Eide and Kathryn Funk and Rodney Kinney and Ziyang Liu and William Merrill and Paul Mooney and Dewey Murdick and Devvret Rishi and Jerry Sheehan and Zhihong Shen and Brandon Stilson and Alex D. Wade and Kuansan Wang and Chris Wilhelm and Boya Xie and Douglas Raymond and Daniel S. Weld and Oren Etzioni and Sebastian Kohlmeier},
    year={2020},
    booktitle={Proceedings of the Workshop on {NLP} for {COVID-19} at {ACL 2020}},
    month = jul,
    url = "https://arxiv.org/abs/2004.10706",
    publisher = "Association for Computational Linguistics”
}
```

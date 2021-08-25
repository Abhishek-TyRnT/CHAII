HindEnCorp 0.5 and HindMonoCorp 0.5 File Formats
================================================

This file describes the file formats of the Hindi-English and Hindi-only
corpora released in 2014 under the names HindEnCorp 0.5 and HindMonoCorp 0.5.

More details about the preparation of the corpora can be found in the paper:

  Ondřej Bojar, Vojtěch Diatka, Pavel Rychlý, Pavel Straňák, Aleš Tamchyna
  and Dan Zeman. HindEnCorp - Hindi-English and Hindi-only Corpus for
  Machine Translation. In Proc. of LREC 2014. Reykjavik, Iceland. ISBN
  978-2-9517408-8-4. ELRA. 2014.

or on the corpora web page:
  http://ufal.mff.cuni.cz/hindencorp

Please cite this paper if you make any use of the corpora. BibTeX citation
format below.


Common Properties
-----------------

All the files are plain text:

- compressed with gzip
- encoded in UTF-8
- with unix line breaks (LF)
- with tab-delimited columns

The monolingual and parallel corpora have different columns.

The actual corpus text is stored in one (monolingual corpus) or two (parallel
corpus) of the columns.


Plaintext vs. Export File Format
--------------------------------

Both the monolingual and the parallel corpus come in a simple plain text format
and in a tokenized, tagged and lemmatized format.

The plaintext format preserves the original tokenization (as much as possible
given the diverse sources included in our corpus).

The 'export' format is tokenized and represents each token as a '|'-delimited
triple of: the word form, the lemma, and part-of-speech tag. If there was the
character '|' (this character is also used instead of the proper Devanagari
Danda in some sources), we escape it as '&pipe;'.

There is exactly the same number of lines in the plaintext and export file
formats.


HindEnCorp Columns
------------------

The files hindencorp05.plaintext.gz and hindencorp05.export.gz each contain the
parallel corpus and differ only in the processing of the corpus texts. The
files have these columns:

- source identifier (where do the segments come from)
- alignment type (number of English segments - number of Hindi segments)
- alignment quality, which is one of the following:
    "manual"  ... for sources that were sentence-aligned manually
    "implied" ... for sources where one side was constructed by translating
                  segment by segment
    float     ... a value somehow reflecting the goodness of the automatic
                  alignment; not really reliable
- English segment or segments
- Hindi segment or segments

Each of the segments field is in the plaintext or export format as described
above.

If there are more than one segments on a line (e.g. for lines with alignment
type 2-1 where there are two English segments), then the segments are delimited
with '<s>' in the text field.

HindMonoCorp Columns
--------------------

The files hindmonocorp05.plaintext.gz and hindmonocorp05.export.gz each contain
the monolingual corpus and differ only in the processing of the corpus text.
Each input segment (usually one Hindi sentence) is stored on a separate line.
The files have these columns:

- source identifier (where does the segment come from)
- segment type, which is one of the following:
    <s>  ... this segment is a sentence from a 'text body'
    <h>  ... this segment comes from a 'headline' (e.g. of an article)
    <a>  ... anything, the source here does not allow to distinguish between
             body and headlines
- Hindi segment


BibTeX Citation for HindEnCorp and HindMonoCorp 0.5
---------------------------------------------------

@InProceedings{hindencorp05:lrec:2014,
  author = {Ond{\v{r}}ej Bojar and Vojt{\v{e}}ch Diatka
            and Pavel Rychl{\'{y}} and Pavel Stra{\v{n}}{\'{a}}k
            and V{\'{\i}}t Suchomel and Ale{\v{s}} Tamchyna and Daniel Zeman},
  title = "{HindEnCorp - Hindi-English and Hindi-only Corpus for Machine
            Translation}",
  booktitle = {Proceedings of the Ninth International Conference on Language
               Resources and Evaluation (LREC'14)},
  year = {2014},
  month = {may},
  date = {26-31},
  address = {Reykjavik, Iceland},
  editor = {Nicoletta Calzolari (Conference Chair) and Khalid Choukri and
     Thierry Declerck and Hrafn Loftsson and Bente Maegaard and Joseph Mariani
     and Asuncion Moreno and Jan Odijk and Stelios Piperidis},
  publisher = {European Language Resources Association (ELRA)},
  isbn = {978-2-9517408-8-4},
  language = {english}
}


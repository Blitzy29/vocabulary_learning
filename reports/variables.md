Variables
==============================

List of available variables in the historical dataset.

| Variable name        | Definition     | Type     |
| :------------- | :---------- | :-----------: |
| nb\_characters\_german | the number of characters in the german translation | numerical |
| nb\_characters\_english | the number of characters in the english translation | numerical |
| nb\_words\_german | the number of words in the german translation | numerical |
| nb\_words\_english | the number of words in the english translation | numerical |
| levenshtein\_distance\_german\_english | the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) between the translations | numerical |
| previous\_score | the previous score for the language asked. It sums the number of successes minus the numbr of fails | numerical |
| previous\_score\_other\_language | the previous score for the other language | numerical |
| previous\_levenshtein\_distance\_guess\_answer | the Levenshtein distance between the previous answer and the translation | numerical |
| previous\_question\_time | the time (in seconds) it took to answer at the previous try | numerical |
| previous\_write\_it\_again\_german | during the previous try, how many times the german translation was written again | numerical |
| previous\_write\_it\_again\_english | during the previous try, how many times the english translation was written again | numerical |
| past\_occurrences\_same\_language | overall, how many times was the word asked in the same language | numerical |
| past\_successes\_same\_language | overall, how many times was the word successfully translated in the same language | numerical |
| past\_fails\_same\_language | overall, how many times was the word wrongly translated in the same language | numerical |
| past\_occurrences\_any\_language | overall, how many times was the word asked in any language | numerical |
| past\_successes\_any\_language | overall, how many times was the word successfully translated in any language | numerical |
| past\_fails\_any\_language | overall, how many times was the word asked in any language | numerical |
| week\_number | the week number of the try | numerical |
| day\_week | the day of week of the try | numerical |
| hour | the hour of the day of the try | numerical |
| nb\_words\_session | the number of words preivous asked during the same session | numerical |
| difficulty\_category | the difficulty category | numerical |
| -- | -- | -- |
| days\_since\_last\_occurrence\_same\_language | the number of days since the last asked translation in the same language | diff_time |
| days\_since\_last\_occurrence\_any\_language | the number of days since the last asked translation in any language | diff_time |
| days\_since\_last\_success\_same\_language | the number of days since the last successful translation in the same language | diff_time |
| days\_since\_last\_success\_any\_language | the number of days since the last successful translation in any language | diff_time |
| days\_since\_first\_occur\_same\_language | the number of days since the first asked translation in the same language | diff_time |
| days\_since\_first\_occur\_any\_language | the number of days since the first asked translation in any language | diff_time |
| -- | -- | -- |
| previous\_result | whther the previous try was correct | boolean |
| previous\_correct\_article | whether the previous try correctly guess the article | boolean |
| previous\_only\_missed\_uppercase | whether the only mistake was an uppercase | boolean |
| previous\_write\_it\_again\_not\_null | whether anything was written again | boolean |
| is\_noun | whether it is a noun | boolean |
| is\_verb | whether it is a verb | boolean |
| previous\_confused\_with\_another\_word | whether we confused it with another word of the proposed dictionary | boolean |
| previous\_confused\_with\_an\_unknown\_word | whether we confused it with another word of the overall dictionary | boolean |
| -- | -- | -- |
| language\_asked | the language asked | categorical |
| previous\_language\_asked | the language asked of the previous try | categorical |
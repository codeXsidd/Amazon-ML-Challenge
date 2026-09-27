import re
import unicodedata


LEGAL_SUFFIXES = re.compile(
    r'\b(inc\.?|incorporated|corp\.?|corporation|llc|l\.l\.c\.?|'
    r'ltd\.?|limited|co\.?|company|plc|pllc|llp|l\.l\.p\.?|'
    r'pvt\.?|private|pub\.?|public|'
    r'gmbh|sa|sarl|sas|srl|ag|bv|nv|'
    r'trust|foundation|association|assn\.?|'
    r'group|holdings?|enterprises?|ventures?|partners?|partnership|'
    r'consultancy|consultants?|solutions?|services?|technologies|tech|'
    r'industries|industrial|international|intl\.?|'
    r'\.com|\.org|\.net|\.in|\.co\.in|\.io)\b',
    re.IGNORECASE
)

ADDR_ABBREVS = {
    'street': 'st', 'road': 'rd', 'avenue': 'ave', 'drive': 'dr',
    'boulevard': 'blvd', 'lane': 'ln', 'court': 'ct', 'place': 'pl',
    'circle': 'cir', 'highway': 'hwy', 'parkway': 'pkwy',
    'terrace': 'ter', 'trail': 'trl', 'way': 'way',
    'apartment': 'apt', 'suite': 'ste', 'building': 'bldg',
    'floor': 'fl', 'unit': 'unit', 'room': 'rm',
    'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
    'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw',
    'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th',
    'fifth': '5th', 'sixth': '6th', 'seventh': '7th', 'eighth': '8th',
    'ninth': '9th', 'tenth': '10th', 'eleventh': '11th', 'twelfth': '12th',
    'mount': 'mt', 'fort': 'ft', 'saint': 'st', 'point': 'pt',
    'township': 'twp', 'village': 'vlg', 'heights': 'hts',
    'crossing': 'xing', 'center': 'ctr', 'square': 'sq',
    'texas': 'tx', 'california': 'ca', 'florida': 'fl',
    'illinois': 'il', 'pennsylvania': 'pa', 'ohio': 'oh',
    'georgia': 'ga', 'michigan': 'mi', 'virginia': 'va',
    'washington': 'wa', 'arizona': 'az', 'massachusetts': 'ma',
    'tennessee': 'tn', 'indiana': 'in', 'missouri': 'mo',
    'maryland': 'md', 'wisconsin': 'wi', 'colorado': 'co',
    'minnesota': 'mn', 'alabama': 'al', 'louisiana': 'la',
    'kentucky': 'ky', 'oregon': 'or', 'oklahoma': 'ok',
    'connecticut': 'ct', 'iowa': 'ia', 'mississippi': 'ms',
    'arkansas': 'ar', 'kansas': 'ks', 'utah': 'ut',
    'nevada': 'nv', 'new mexico': 'nm', 'nebraska': 'ne',
    'idaho': 'id', 'hawaii': 'hi', 'maine': 'me',
    'montana': 'mt', 'delaware': 'de', 'south dakota': 'sd',
    'north dakota': 'nd', 'alaska': 'ak', 'vermont': 'vt',
    'wyoming': 'wy', 'new hampshire': 'nh', 'new jersey': 'nj',
    'new york': 'ny', 'north carolina': 'nc', 'south carolina': 'sc',
    'west virginia': 'wv', 'rhode island': 'ri',
    'district of columbia': 'dc',
    'karnataka': 'ka', 'maharashtra': 'mh', 'telangana': 'ts',
    'rajasthan': 'rj', 'tamil nadu': 'tn', 'uttar pradesh': 'up',
    'madhya pradesh': 'mp', 'gujarat': 'gj', 'kerala': 'kl',
    'andhra pradesh': 'ap', 'west bengal': 'wb', 'punjab': 'pb',
    'haryana': 'hr', 'bihar': 'br', 'odisha': 'od',
    'jharkhand': 'jh', 'chhattisgarh': 'cg', 'assam': 'as',
}


def normalize_unicode(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text


def normalize_name(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = normalize_unicode(text)
    text = text.lower().strip()
    text = text.replace('&', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = LEGAL_SUFFIXES.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_name_keep_suffix(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = normalize_unicode(text)
    text = text.lower().strip()
    text = text.replace('&', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_name_tokens(normalized_name):
    if not normalized_name:
        return set()
    tokens = set(normalized_name.split())
    tokens.discard('')
    return tokens


def get_sorted_name(normalized_name):
    if not normalized_name:
        return ''
    return ' '.join(sorted(normalized_name.split()))


def normalize_address(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = normalize_unicode(text)
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    result = []
    for w in words:
        result.append(ADDR_ABBREVS.get(w, w))
    text = ' '.join(result)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_address_tokens(normalized_addr):
    if not normalized_addr:
        return set()
    tokens = set(normalized_addr.split())
    tokens.discard('')
    return tokens


def get_address_numbers(normalized_addr):
    if not normalized_addr:
        return set()
    return set(re.findall(r'\d+', normalized_addr))


def get_compact_name(name):
    if not name:
        return ''
    return re.sub(r'\s+', '', name)


def get_name_char_ngrams(name, n=3):
    if not name or len(name) < n:
        return set()
    compact = get_compact_name(name)
    return {compact[i:i+n] for i in range(len(compact) - n + 1)}

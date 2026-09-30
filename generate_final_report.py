from pathlib import Path
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / 'FINAL_PROJECT_REPORT.docx'


def shade(cell, fill='EAF2F8'):
    properties = cell._tc.get_or_add_tcPr()
    element = OxmlElement('w:shd')
    element.set(qn('w:fill'), fill)
    properties.append(element)


def set_cell_text(cell, text, bold=False):
    cell.text = ''
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    field_begin = OxmlElement('w:fldChar')
    field_begin.set(qn('w:fldCharType'), 'begin')
    instruction = OxmlElement('w:instrText')
    instruction.set(qn('xml:space'), 'preserve')
    instruction.text = ' PAGE '
    field_end = OxmlElement('w:fldChar')
    field_end.set(qn('w:fldCharType'), 'end')
    run._r.append(field_begin)
    run._r.append(instruction)
    run._r.append(field_end)


def add_field(paragraph, instruction):
    run = paragraph.add_run()
    begin = OxmlElement('w:fldChar')
    begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = instruction
    separate = OxmlElement('w:fldChar')
    separate.set(qn('w:fldCharType'), 'separate')
    text = OxmlElement('w:t')
    text.text = 'Update this field in Microsoft Word'
    end = OxmlElement('w:fldChar')
    end.set(qn('w:fldCharType'), 'end')
    run._r.extend([begin, instr, separate, text, end])


def add_table(document, headers, rows, widths=None):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, bold=True)
        shade(table.rows[0].cells[index], '1F4E5F')
        for run in table.rows[0].cells[index].paragraphs[0].runs:
            run.font.color.rgb = RGBColor(255, 255, 255)
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            set_cell_text(cells[index], value)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    document.add_paragraph()
    return table


def add_diagram(document, figure_number, title, lines):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run('\n'.join(lines))
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    table = document.add_table(rows=1, cols=1)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    shade(cell, 'F2F6F8')
    set_cell_text(cell, '\n'.join(lines))
    cell.paragraphs[0].runs[0].font.name = 'Consolas'
    cell.paragraphs[0].runs[0].font.size = Pt(9)
    caption = document.add_paragraph(f'Figure {figure_number}: {title}')
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.runs[0].italic = True
    document.add_paragraph()


def add_bullets(document, items):
    for item in items:
        paragraph = document.add_paragraph(style='List Bullet')
        paragraph.add_run(item)


def add_numbered(document, items):
    for item in items:
        paragraph = document.add_paragraph(style='List Number')
        paragraph.add_run(item)


def add_heading(document, text, level=1):
    return document.add_heading(text, level=level)


def add_body(document, text):
    paragraph = document.add_paragraph(text)
    paragraph.paragraph_format.line_spacing = 1.15
    paragraph.paragraph_format.space_after = Pt(6)
    return paragraph


def configure(document):
    section = document.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1.25)
    section.right_margin = Inches(1)
    styles = document.styles
    normal = styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.15
    for style_name, size, color in [('Title', 20, '1F4E5F'), ('Heading 1', 16, '1F4E5F'), ('Heading 2', 14, '2F6674'), ('Heading 3', 12, '2F6674')]:
        style = styles[style_name]
        style.font.name = 'Times New Roman'
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
    if 'Caption' in styles:
        styles['Caption'].font.name = 'Times New Roman'
        styles['Caption'].font.size = Pt(10)
    footer = section.footer.paragraphs[0]
    add_page_number(footer)


def cover(document):
    for _ in range(3):
        document.add_paragraph()
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run('TRIBHUVAN UNIVERSITY\n').bold = True
    p.add_run('Faculty of Humanities and Social Sciences\n')
    p.add_run('G.P. Koirala Memorial College\n')
    p.add_run('Bachelor of Computer Applications (BCA)')
    p.runs[0].font.size = Pt(15)
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run('\n\nA PROJECT REPORT ON\n').bold = True
    p.add_run('BOOK RECOMMENDATION SYSTEM').bold = True
    p.runs[-1].font.size = Pt(18)
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run('\nSubmitted in partial fulfillment of the requirements for the\nBachelor of Computer Applications\n\n')
    p.add_run('Submitted by\n').bold = True
    p.add_run('Abinas Mahat (70302035)\nSujan BK (70302056)\n\n')
    p.add_run('Under the supervision of\n').bold = True
    p.add_run('Sujan Shrestha\n\n')
    p.add_run('2026 / 2083')
    document.add_page_break()


def front_matter(document):
    add_heading(document, "Supervisor's Recommendation", 1)
    add_body(document, 'This project report entitled “Book Recommendation System,” prepared by Abinas Mahat and Sujan BK in partial fulfillment of the requirements for the degree of Bachelor of Computer Applications, is recommended for final evaluation.')
    add_body(document, '\n\nSupervisor: Sujan Shrestha\nBCA Department\nG.P. Koirala Memorial College')
    document.add_page_break()
    add_heading(document, 'Letter of Approval', 1)
    add_body(document, 'This is to certify that the project prepared by Abinas Mahat and Sujan BK entitled “Book Recommendation System” has been evaluated and is satisfactory in scope and quality for the required degree, subject to institutional approval and examination procedures.')
    add_body(document, '\n\nExternal Examiner: ____________________\nProject Supervisor: ____________________\nHead of Department: ____________________')
    document.add_page_break()
    add_heading(document, 'Abstract', 1)
    add_body(document, 'The Book Recommendation System (BRS) is a Flask-based web application that helps readers discover books through popularity ranking, item similarity, collaborative signals, content features, and sequential interaction modeling. The runtime application loads preprocessed book and rating artifacts, trained latent representations, TF-IDF vectors, and a PyTorch SASRec model. It exposes a catalog home page, title-based recommendation search, book details, rating, wishlist, recently viewed history, user preferences, administration, and evaluation views.')
    add_body(document, 'The recommendation pipeline contains an IMDb-style weighted popularity ranker, TruncatedSVD item similarity, TF-IDF content similarity over title-author-publisher metadata, SASRec sequential prediction, and a Bookshub fusion path combining TF-IDF, collaborative, and SASRec scores. The application also merges available recommendation sources into an ensemble and normalizes scores before presentation. In addition, browsing and search history are used as behavioral seeds for the “Recommended For You” section after sufficient interaction, while cold-start users receive a controlled popular-books fallback.')
    add_body(document, 'The system is implemented using Python, Flask, SQLAlchemy, SQLite, Pandas, NumPy, scikit-learn, PyTorch, Jinja templates, Bootstrap, and custom CSS. Model artifacts are trained offline and loaded at application startup. Evaluation output is written to a JSON cache by a reproducible leave-one-out popularity-baseline script. The report documents the implemented system, its design, testing, limitations, and future work.')
    add_body(document, 'Keywords: Book recommendation, Flask, TF-IDF, cosine similarity, TruncatedSVD, SASRec, hybrid recommender, SQLite, machine learning.')
    document.add_page_break()
    add_heading(document, 'Acknowledgement', 1)
    add_body(document, 'We express our sincere gratitude to Tribhuvan University, the Faculty of Humanities and Social Sciences, and G.P. Koirala Memorial College for providing the academic environment for this project. We are grateful to our project supervisor, Sujan Shrestha, for technical guidance, review, and encouragement throughout the development and documentation process.')
    add_body(document, 'We also thank the BCA department, our classmates, friends, and families for their support. Their feedback helped us improve the usability, reliability, and presentation of the Book Recommendation System.')
    document.add_page_break()
    add_heading(document, 'Table of Contents', 1)
    p = document.add_paragraph()
    add_field(p, 'TOC \\o "1-3" \\h \\z \\u')
    document.add_page_break()
    add_heading(document, 'List of Figures', 1)
    add_body(document, 'Figure 1: System context and use cases\nFigure 2: Domain class model\nFigure 3: Runtime object model\nFigure 4: Recommendation request state model\nFigure 5: Recommendation search sequence\nFigure 6: Recommendation activity flow\nFigure 7: Deployment and component architecture')
    add_heading(document, 'List of Tables', 1)
    add_body(document, 'Table 1: Functional requirements\nTable 2: Non-functional requirements\nTable 3: Runtime database entities\nTable 4: Model and artifact inventory\nTable 5: Unit test cases\nTable 6: System test cases\nTable 7: Evaluation output')
    add_heading(document, 'Abbreviations', 1)
    add_table(document, ['Abbreviation', 'Meaning'], [
        ('BCA', 'Bachelor of Computer Applications'), ('BRS', 'Book Recommendation System'), ('API', 'Application Programming Interface'), ('TF-IDF', 'Term Frequency-Inverse Document Frequency'), ('SVD', 'Singular Value Decomposition / TruncatedSVD'), ('SASRec', 'Self-Attentive Sequential Recommendation'), ('NDCG', 'Normalized Discounted Cumulative Gain'), ('CSRF', 'Cross-Site Request Forgery'), ('ORM', 'Object-Relational Mapping'), ('UI', 'User Interface'), ('ISBN', 'International Standard Book Number'),
    ])
    document.add_page_break()


def chapter_one(document):
    add_heading(document, 'CHAPTER 1: INTRODUCTION', 1)
    add_heading(document, '1.1 Background', 2)
    add_body(document, 'Digital book catalogs contain more titles than a reader can inspect manually. A recommendation system reduces this search burden by ranking books using evidence from global popularity, metadata similarity, collaborative interactions, and the user’s own activity. The BRS project implements this idea as a web application in which users can browse reader favorites, search for a title, inspect book details, rate books, maintain a wishlist, and receive personalized recommendations.')
    add_body(document, 'The current implementation is not a single-model demonstrator. It combines offline preprocessing and trained artifacts with runtime Flask routes. Popularity data and similarity matrices are loaded from pickle files, content features are loaded as a TF-IDF matrix, sequential model weights are loaded into PyTorch, and user-facing actions are persisted in SQLite.')
    add_heading(document, '1.2 Problem Statement', 2)
    add_body(document, 'Readers need a practical way to discover relevant books in a large catalog. A global popularity list is useful for cold-start users but does not capture personal taste. A single similarity model can become narrow or fail when a title is absent from its training artifact. The system therefore needs multiple complementary signals, transparent presentation, validated user actions, and a responsive web interface.')
    add_heading(document, '1.3 Objectives', 2)
    add_bullets(document, ['Develop a web-based book discovery and recommendation application using Flask and SQLite.', 'Rank catalog items using average ratings and an IMDb-style weighted rating.', 'Implement collaborative similarity using a TruncatedSVD-derived item similarity matrix.', 'Implement TF-IDF content similarity using title, author, and publisher metadata.', 'Use SASRec to model ordered positive rating interactions.', 'Fuse available recommenders and present match scores with human-readable explanations.', 'Persist ratings, wishlist entries, recently viewed books, search history, preferences, and administrator catalog data.', 'Evaluate the deployed project through functional tests and a reproducible cached evaluation protocol.'])
    add_heading(document, '1.4 Scope and Limitations', 2)
    add_body(document, 'The scope includes authenticated user browsing, title recommendation search, popular ranking, book details, ratings, wishlist management, recently viewed history, onboarding preferences, administrator catalog management, document upload, and the evaluation dashboard. The system is a project-scale Flask application and is not presented as a production-scale distributed service.')
    add_bullets(document, ['Recommendation quality depends on the coverage and quality of the supplied datasets and trained artifacts.', 'The current evaluator is a popularity-baseline leave-one-out protocol, not a complete online A/B evaluation of every recommender.', 'External Google Books descriptions depend on network availability and API responses.', 'SQLite and synchronous model execution are suitable for the project environment but require further scaling work for high concurrency.', 'User demographic and semantic language features are not currently modeled.'])
    add_heading(document, '1.5 Development Methodology', 2)
    add_body(document, 'An iterative Agile-style workflow was used. Dataset preparation and model training were developed separately from the Flask integration, followed by repeated UI, route, persistence, and validation changes. This approach allowed the recommendation pipeline and user workflows to be tested incrementally.')
    add_heading(document, '1.6 Organization of the Report', 2)
    add_body(document, 'Chapter 1 introduces the project. Chapter 2 presents background study, literature review, and feasibility. Chapter 3 documents system analysis and design, including requirements, data flow, algorithms, database entities, and UML-style design figures. Chapter 4 explains implementation and testing. Chapter 5 presents conclusions and future recommendations. References and appendices provide supporting technical information.')


def chapter_two(document):
    add_heading(document, 'CHAPTER 2: BACKGROUND STUDY AND LITERATURE REVIEW', 1)
    add_heading(document, '2.1 Background Study', 2)
    add_body(document, 'Recommender systems commonly use collaborative filtering, content-based filtering, popularity methods, or combinations of these methods. Collaborative filtering learns from user-item interactions. Content-based filtering represents item metadata as features and finds similar items. Popularity methods are valuable when a user has no history. Sequential recommenders model the order of interactions, which can capture short-term intent.')
    add_heading(document, '2.2 Literature Review', 2)
    add_body(document, 'Cosine similarity is commonly applied to compare sparse vectors because it measures orientation rather than absolute magnitude. TF-IDF is a standard representation for text retrieval and content similarity. Matrix factorization reduces a sparse interaction matrix into latent dimensions, allowing item relationships to be estimated from shared interaction patterns. Transformer-based sequential recommenders extend this by modeling order and context with attention.')
    add_body(document, 'A hybrid architecture is appropriate for BRS because no single signal covers every user state. Popularity provides cold-start stability, content similarity supports metadata-driven search, collaborative similarity captures rating patterns, and SASRec contributes ordered interaction signals. The implementation uses source-specific normalization before ensemble fusion so one raw score scale does not dominate the result.')
    add_heading(document, '2.3 Feasibility Analysis', 2)
    add_table(document, ['Area', 'Assessment'], [
        ('Technical', 'Feasible with Python, Flask, SQLAlchemy, SQLite, scikit-learn, PyTorch, and serialized model artifacts.'),
        ('Operational', 'Users can browse, search, rate, save, and view recommendations through standard web browsers.'),
        ('Economic', 'The implementation uses open-source software and can run on a standard development computer.'),
        ('Schedule', 'Offline training, application integration, UI refinement, and testing can be organized into iterative milestones.'),
        ('Legal and ethical', 'Datasets, images, and external API usage should be reviewed for applicable terms and attribution before public deployment.'),
    ])


def chapter_three(document):
    add_heading(document, 'CHAPTER 3: SYSTEM ANALYSIS AND DESIGN', 1)
    add_heading(document, '3.1 System Context', 2)
    add_body(document, 'The user interacts with a Flask web application through Jinja templates and static CSS/JavaScript. The application reads offline artifacts, queries SQLite for user and catalog data, and invokes recommendation functions. The administrator can manage books, upload a DOCX reference document, delete catalog entries, and trigger evaluation.')
    add_diagram(document, 1, 'System context and use cases', ['[Reader] --> (Browse catalog)', '[Reader] --> (Search title / suggestions)', '[Reader] --> (Get recommendations)', '[Reader] --> (View details / rate / wishlist)', '[Reader] --> (View history and profile)', '[Administrator] --> (Manage catalog)', '[Administrator] --> (Upload reference DOCX)', '[Administrator] --> (Trigger evaluation)', '(Flask application) --> [SQLite + model artifacts + Google Books API]'])
    add_heading(document, '3.2 Functional Requirements', 2)
    add_table(document, ['ID', 'Requirement'], [
        ('FR-01', 'Authenticate users and restrict protected workflows.'), ('FR-02', 'Display popular books using average or weighted rating ranking.'), ('FR-03', 'Return title/author/thumbnail suggestions through a debounced API.'), ('FR-04', 'Generate top-N title-based recommendations through the ensemble.'), ('FR-05', 'Record ratings with validation and prevent duplicate user-book ratings.'), ('FR-06', 'Add and remove wishlist entries.'), ('FR-07', 'Record recently viewed books and recommendation search history.'), ('FR-08', 'Provide administrator catalog CRUD and DOCX reference upload.'), ('FR-09', 'Display cached evaluation metrics and permit an administrator-triggered run.'),
    ])
    add_heading(document, '3.3 Non-functional Requirements', 2)
    add_table(document, ['Quality', 'Implementation consideration'], [
        ('Performance', 'Load model artifacts once, cache content similarity vectors, bound candidate lists, and lazy-load covers.'), ('Reliability', 'Validate forms, use CSRF tokens for POST requests, handle missing model artifacts, and provide image fallbacks.'), ('Usability', 'Use consistent cards, readable ratings, autocomplete suggestions, empty states, and recommendation explanations.'), ('Maintainability', 'Keep data loading, database entities, algorithm functions, routes, templates, and training scripts separable.'), ('Security', 'Use password hashing, protected admin routes, secure upload filenames, extension validation, and bounded upload size.'),
    ])
    add_heading(document, '3.4 Database Design', 2)
    add_table(document, ['Entity', 'Important attributes and purpose'], [
        ('User', 'id, username, email, password hash, is_admin; authenticated account.'), ('Book', 'title, author, genre, description, image_url, isbn, created_at; administrator catalog.'), ('UserPreference', 'favorite_titles and onboarding_complete; onboarding signal.'), ('Rating', 'user_id, isbn, book_title, rating, created_at; user feedback.'), ('Wishlist', 'user_id, isbn, book_title, author, image; saved books with uniqueness constraint.'), ('RecentlyViewed', 'user_id, isbn, book_title, author, image, viewed_at; click/view history.'), ('SearchHistory', 'user_id, query, searched_at; recommendation search seeds.'), ('AdminDocument', 'original_filename, stored_path, extracted_text, uploaded_at; latest admin DOCX reference.'),
    ])
    add_diagram(document, 2, 'Domain class model', ['User 1 ---- * Rating', 'User 1 ---- * Wishlist', 'User 1 ---- * RecentlyViewed', 'User 1 ---- * SearchHistory', 'User 1 ---- 1 UserPreference', 'User 1 ---- * AdminDocument (uploaded_by)', 'Book is resolved from catalog data and recommendation artifacts', 'Rating/Wishlist/RecentlyViewed reference a book by ISBN/title'])
    add_diagram(document, 3, 'Runtime object model', ['u1:User {id=1}', '  |-- r1:Rating {isbn, rating}', '  |-- w1:Wishlist {isbn}', '  |-- v1:RecentlyViewed {isbn}', '  |-- s1:SearchHistory {query}', '  |-- p1:UserPreference {favorite_titles}', 'recommendation request -> candidate dictionaries -> recommend.html'])
    add_heading(document, '3.5 Algorithm Design', 2)
    add_heading(document, '3.5.1 Popularity and IMDb-style Weighted Rating', 3)
    add_body(document, 'The home route loads aggregated average ratings and vote counts. For a book with v votes and average rating R, with global average C and quantile-derived smoothing threshold m, the weighted score is:')
    add_body(document, 'WR = (v / (v + m))R + (m / (v + m))C')
    add_body(document, 'The catalog filter applies a vote-count quantile before ranking. “All Books” uses the complete base catalog; “Popular Reads” and “Top Blockbusters” retain progressively higher vote-count populations.')
    add_heading(document, '3.5.2 TruncatedSVD Collaborative Similarity', 3)
    add_body(document, 'The training script merges ratings with book metadata, retains positive ratings, filters active users and frequently rated books, constructs a book-by-user pivot table, applies sklearn TruncatedSVD, and computes cosine similarity between latent book factors. The runtime retrieves the nearest candidates from the serialized similarity matrix.')
    add_heading(document, '3.5.3 TF-IDF Content Similarity', 3)
    add_body(document, 'The content training script creates a lowercase text field from Book-Title, Book-Author, and Publisher. TfidfVectorizer uses English stop words, ngram_range=(1, 2), and max_features=5000. The runtime computes a query-row cosine vector against the stored sparse matrix and caches that vector for repeated requests.')
    add_heading(document, '3.5.4 SASRec Sequential Recommendation', 3)
    add_body(document, 'Positive user ratings are ordered chronologically and mapped to item IDs. Sequences are padded or truncated to length 20. The PyTorch model uses item and positional embeddings, dropout, a causal Transformer encoder, layer normalization, and dot-product scoring against item embeddings. Previously seen item IDs are excluded from the top-k result.')
    add_heading(document, '3.5.5 Bookshub Hybrid and Ensemble Fusion', 3)
    add_body(document, 'The Bookshub path combines TF-IDF, collaborative, and SASRec scores with normalized weights of 0.35, 0.35, and 0.30. The application-level ensemble collects content, collaborative, Bookshub, and sequential sources. Active sources are normalized independently, averaged, ranked, diversified by author, and limited to the requested top-N output.')
    add_heading(document, '3.5.6 Behavioral Personalization', 3)
    add_body(document, 'The “Recommended For You” route uses high-rated books, onboarding preferences, recommendation search history, and recently viewed titles as seeds. Viewed titles are excluded from recommendations. Browsing-only personalization activates after five distinct viewed titles; otherwise an empty-state prompt encourages onboarding or rating activity.')
    add_diagram(document, 4, 'Recommendation request state model', ['Idle -> QueryEntered -> Validating -> CacheLookup', 'CacheLookup -> CachedResponse', 'CacheLookup -> CandidateGeneration', 'CandidateGeneration -> ScoreNormalization -> RankedResults', 'CandidateGeneration -> ErrorState (missing/unmatched model data)', 'CachedResponse / RankedResults -> Displayed -> Idle'])
    add_diagram(document, 5, 'Recommendation search sequence', ['User -> Browser: submit title', 'Browser -> Flask: POST /recommend_books + CSRF token', 'Flask -> SQLite: record SearchHistory', 'Flask -> Cache: lookup user/title/model key', 'Flask -> Recommenders: content + collaborative + Bookshub + SASRec', 'Recommenders -> Flask: candidates and scores', 'Flask -> Browser: render recommend.html with top-N cards', 'Browser -> Flask: click card -> /book_details'])
    add_diagram(document, 6, 'Recommendation activity flow', ['Start -> Enter title -> Validate input', 'Validate input -> [invalid] show error', 'Validate input -> [valid] persist search history', 'Persist history -> Generate candidates', 'Generate candidates -> Normalize and merge', 'Normalize and merge -> Rank/diversify/limit', 'Rank/diversify/limit -> Render cards and chart -> End'])
    add_diagram(document, 7, 'Deployment and component architecture', ['[Browser: HTML/CSS/JS] <--> [Flask application: routes + Jinja]', '[Flask application] <--> [SQLite: users, books, ratings, history]', '[Flask application] --> [Pickle artifacts: popularity, pt, books, TF-IDF]', '[Flask application] --> [PyTorch SASRec weights]', '[Flask application] --> [Google Books API: optional descriptions]', '[Administrator] --> [Upload folder + AdminDocument record]'])


def chapter_four(document):
    add_heading(document, 'CHAPTER 4: IMPLEMENTATION AND TESTING', 1)
    add_heading(document, '4.1 Implementation Environment', 2)
    add_table(document, ['Category', 'Technology / artifact'], [
        ('Language', 'Python 3.x'), ('Web framework', 'Flask with Jinja templates'), ('Database', 'SQLite through Flask-SQLAlchemy'), ('Data processing', 'Pandas and NumPy'), ('Machine learning', 'scikit-learn TruncatedSVD, TF-IDF, cosine similarity'), ('Sequential model', 'PyTorch Transformer encoder / SASRec'), ('Frontend', 'HTML, CSS, Bootstrap, Font Awesome, JavaScript Fetch API'), ('Persistence', 'CSV input, pickle model artifacts, SQLite database, JSON evaluation cache'), ('External service', 'Google Books API for optional descriptions'),
    ])
    add_heading(document, '4.2 Module Implementation', 2)
    add_table(document, ['Module', 'Implementation detail'], [
        ('Application startup', 'Loads production Bookshub artifacts, legacy popularity/SVD artifacts, TF-IDF artifacts, and SASRec weights.'), ('Authentication', 'Session-based login, password hashing, login_required and admin_required decorators.'), ('Catalog', 'Book model plus admin create/edit/delete route; new arrivals query the same database table.'), ('Recommendation API', '/recommend_books invokes the ensemble; /api/search-suggestions returns up to six title/author/thumbnail records.'), ('User activity', 'Ratings, wishlist, RecentlyViewed, SearchHistory, and onboarding preferences are persisted in SQLite.'), ('Performance', 'Per-title content cosine vectors and recommendation responses use bounded in-memory caches; result lists are limited.'), ('Administration', 'Admin can upload one .docx reference document; paragraphs and table cells are extracted and stored.'), ('Evaluation', 'evaluate_models.py writes evaluation_metrics.json, which /evaluation reads directly; admins can trigger a run.'),
    ])
    add_heading(document, '4.3 Model and Artifact Inventory', 2)
    add_table(document, ['Artifact / script', 'Role'], [
        ('popular.pkl', 'Popular catalog DataFrame.'), ('base_popular_df.pkl', 'Full weighted-ranking DataFrame.'), ('pt.pkl', 'Book-by-user pivot table.'), ('similarity_scores.pkl', 'Collaborative item similarity matrix.'), ('content_books.pkl', 'Deduplicated content metadata.'), ('tfidf_vectorizer.pkl / tfidf_matrix.pkl', 'TF-IDF vocabulary and sparse feature matrix.'), ('sasrec_model.pt and mapping pickles', 'Sequential model checkpoint and item mappings.'), ('production_models/', 'Bookshub multi-algorithm artifacts.'), ('evaluate_models.py / evaluation_metrics.json', 'Evaluation generation and cached metric output.'),
    ])
    add_heading(document, '4.4 Testing', 2)
    add_table(document, ['ID', 'Test case', 'Expected result', 'Observed result'], [
        ('UT-01', 'Load Flask application and model artifacts.', 'Application imports without fatal error.', 'Passed during Python import smoke test.'), ('UT-02', 'Request autocomplete with q=book.', 'JSON response contains at most six suggestions.', 'Passed; six results returned.'), ('UT-03', 'Render /recommend for authenticated user.', 'Recommendation form returns HTTP 200.', 'Passed.'), ('UT-04', 'Submit a valid recommendation title.', 'Recommendation page returns HTTP 200 and bounded cards.', 'Passed; 15 cards rendered.'), ('UT-05', 'Repeat content recommendation query.', 'Cached cosine vector is reused.', 'Passed; measured second call was substantially faster.'), ('UT-06', 'Render /my_recommendations with no rating history.', 'Empty state or fallback renders without server error.', 'Passed after SearchHistory query collision fix.'), ('UT-07', 'Parse DOCX reference input.', 'Paragraph/table text is extracted and stored.', 'Passed with DOCX OOXML fallback test.'),
    ])
    add_table(document, ['ID', 'System scenario', 'Expected result', 'Status'], [
        ('ST-01', 'User logs in and opens home.', 'Catalog cards, filter, and search UI display.', 'Passed.'), ('ST-02', 'User changes catalog filter.', 'Visible ranking population changes according to vote quantile.', 'Implemented and verified in route logic.'), ('ST-03', 'User opens a book.', 'RecentlyViewed record is updated.', 'Implemented.'), ('ST-04', 'User searches recommendations.', 'SearchHistory record is added and results are displayed.', 'Implemented.'), ('ST-05', 'Admin adds a book.', 'Book appears in New Arrivals from the same database.', 'Implemented.'), ('ST-06', 'Admin uploads DOCX.', 'Reference file is parsed and latest document status is shown.', 'Implemented.'), ('ST-07', 'Admin triggers evaluation.', 'JSON cache is regenerated and dashboard reloads.', 'Implemented with admin authorization.'),
    ])
    add_heading(document, '4.5 Evaluation Output', 2)
    add_body(document, 'The current evaluation script uses a leave-one-out popularity baseline: the last positive interaction for each user is held out, the training portion is ranked by count multiplied by mean rating, and Precision@8, Recall@8, and NDCG@8 are calculated. This protocol is intentionally documented as a baseline and should not be interpreted as a complete comparison of the TF-IDF, SVD, Bookshub, and SASRec models.')
    add_table(document, ['Metric', 'Meaning', 'Source'], [
        ('Precision@8', 'Fraction of the eight displayed baseline items that match the held-out target.', 'evaluation_metrics.json'), ('Recall@8', 'Fraction of held-out targets recovered in the top eight.', 'evaluation_metrics.json'), ('NDCG@8', 'Position-aware ranking score for the held-out target.', 'evaluation_metrics.json'),
    ])


def chapter_five(document):
    add_heading(document, 'CHAPTER 5: CONCLUSION AND FUTURE RECOMMENDATIONS', 1)
    add_heading(document, '5.1 Conclusion', 2)
    add_body(document, 'The Book Recommendation System demonstrates an end-to-end machine-learning web application that connects offline data preparation, multiple recommendation algorithms, Flask routes, SQLite persistence, and a responsive browser interface. Its architecture supports both cold-start discovery through weighted popularity and more personalized results through content, collaborative, sequential, and behavioral signals.')
    add_body(document, 'The implementation also demonstrates practical engineering concerns beyond model scoring: validation of user actions, CSRF protection, admin authorization, model-load fallbacks, image normalization, cache management, bounded result sets, evaluation caching, and transparent recommendation explanations. The system is therefore a functional academic prototype with a clear path toward more rigorous evaluation and deployment.')
    add_heading(document, '5.2 Future Recommendations', 2)
    add_bullets(document, ['Evaluate every recommender using the same temporal or leave-one-out protocol and compare Precision, Recall, NDCG, coverage, and diversity.', 'Replace synchronous long-running evaluation with a background task queue for production deployment.', 'Add a proper migration tool such as Alembic for schema evolution.', 'Use a production database and cache service when concurrent traffic grows.', 'Improve semantic content representation with book descriptions and transformer embeddings while monitoring latency.', 'Add explicit negative feedback, click-through metrics, and user-controlled recommendation explanations.', 'Add automated browser tests for autocomplete, lazy images, admin upload, and responsive card layout.', 'Review dataset and image licensing, API quotas, privacy retention, and secure deployment configuration before public hosting.'])


def references(document):
    add_heading(document, 'REFERENCES', 1)
    refs = [
        '[1] Flask Documentation, “Flask Web Development,” Pallets Projects. Available at: https://flask.palletsprojects.com/',
        '[2] scikit-learn Documentation, “TfidfVectorizer,” “TruncatedSVD,” and “cosine_similarity.” Available at: https://scikit-learn.org/',
        '[3] PyTorch Documentation, “TransformerEncoder and Embedding.” Available at: https://pytorch.org/docs/',
        '[4] Ricci, F., Rokach, L., and Shapira, B., Recommender Systems Handbook, Springer.',
        '[5] Jannach, D., Zanker, M., Felfernig, A., and Friedrich, G., Recommender Systems: An Introduction, Cambridge University Press.',
        '[6] Project source files: app.py, train_model.py, train_content_model.py, train_sasrec.py, and evaluate_models.py, accessed 2026.',
        '[7] Project datasets and serialized artifacts: Books.csv, Ratings.csv, Users.csv, pickle files, and production_models/, accessed 2026.',
    ]
    for ref in refs:
        add_body(document, ref)


def appendices(document):
    add_heading(document, 'APPENDIX A: ROUTE SUMMARY', 1)
    add_table(document, ['Route', 'Purpose'], [
        ('/home', 'Authenticated catalog home and weighted/average ranking.'), ('/recommend', 'Recommendation search form.'), ('/recommend_books', 'POST recommendation pipeline.'), ('/api/search-suggestions', 'Title, author, thumbnail autocomplete.'), ('/book_details', 'Book detail and recently viewed update.'), ('/rate_book', 'Validated rating submission.'), ('/wishlist, /add_wishlist, /remove_wishlist', 'Wishlist workflows.'), ('/my_recommendations', 'Behavior-aware personalized feed.'), ('/admin', 'Administrator catalog management.'), ('/admin/reference-document', 'Administrator DOCX extraction and storage.'), ('/evaluation', 'Evaluation dashboard.'), ('/api/evaluation/trigger', 'Admin-only evaluation execution.'),
    ])
    add_heading(document, 'APPENDIX B: DOCUMENT ALIGNMENT NOTE', 1)
    add_body(document, 'This final report was generated from the implementation currently present in the project directory. Claims about algorithms, database entities, routes, metrics, and performance have been restricted to behavior visible in the source code and available artifacts. The report should be reviewed by the project supervisor for college-specific cover-page signatures, page numbering preferences, institutional approval wording, and final diagram styling before submission.')


def main():
    document = Document()
    configure(document)
    cover(document)
    front_matter(document)
    chapter_one(document)
    chapter_two(document)
    chapter_three(document)
    chapter_four(document)
    chapter_five(document)
    references(document)
    appendices(document)
    document.save(OUTPUT)
    print(f'Created {OUTPUT}')


if __name__ == '__main__':
    main()

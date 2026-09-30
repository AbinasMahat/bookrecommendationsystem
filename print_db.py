from app import app, db, Book, User, Rating, Wishlist, RecentlyViewed, SearchHistory
with app.app_context():
    print('\n================ USER DETAILS & PASSWORD HASHES ================')
    for u in User.query.all(): print(f'ID: {u.id} | User: {u.username} | Email: {u.email} | Admin: {u.is_admin}\n   -> Hashed Password: {u.password}\n')
    print('================ WISHLIST ITEMS ================')
    for w in Wishlist.query.all(): print(f'User ID: {w.user_id} | Book: {w.book_title} | ISBN: {w.isbn}')
    print('\n================ RECENTLY VIEWED BOOKS ================')
    for rv in RecentlyViewed.query.all(): print(f'User ID: {rv.user_id} | Book: {rv.book_title} | Viewed At: {rv.viewed_at}')
    print('\n================ SEARCH HISTORY ================')
    for sh in SearchHistory.query.all(): print(f'User ID: {sh.user_id} | Query: {sh.query} | Searched At: {sh.searched_at}')

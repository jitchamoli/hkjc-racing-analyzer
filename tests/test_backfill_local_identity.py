"""Local card backfills must not relabel overseas horse-history rows."""
import pytest

from data.database import HorseDatabase


@pytest.fixture
def db(tmp_path):
    database = HorseDatabase(db_path=str(tmp_path / 'history.db'))
    yield database
    database.close()


def insert(db, name, venue, number=None):
    db.conn.execute(
        'INSERT INTO race_results (race_date,race_number,horse_name,venue,'
        'horse_number,distance,class_type) VALUES (?,?,?,?,?,?,?)',
        ('2024/06/02', 1, name, venue, number, 1600, 'Group 1'))
    db.conn.commit()


def snapshot(db, name):
    return db.conn.execute('SELECT * FROM race_results WHERE horse_name=?',
                           (name,)).fetchone()


def card(horses, venue='Sha Tin'):
    return [{'race_number': 1, 'venue': venue, 'distance': 2000,
             'class_type': 'Class 5', 'horses': horses}]


@pytest.mark.parametrize('venue', [None, '', 'Overseas', 'Tokyo'])
def test_unlisted_overseas_runner_is_untouched(db, venue):
    insert(db, 'ROMANTIC WARRIOR', venue)
    insert(db, 'HAPPY HERO', 'Sha Tin', 7)
    before = tuple(snapshot(db, 'ROMANTIC WARRIOR'))
    db.backfill_card_fields('2024/06/02', card([{'name': 'HAPPY HERO', 'number': 7}]))
    assert tuple(snapshot(db, 'ROMANTIC WARRIOR')) == before
    assert snapshot(db, 'HAPPY HERO')['distance'] == 2000


def test_empty_card_cannot_bulk_relabel_race(db):
    insert(db, 'ROMANTIC WARRIOR', None)
    before = tuple(snapshot(db, 'ROMANTIC WARRIOR'))
    assert db.backfill_card_fields('2024/06/02', card([])) == 0
    assert tuple(snapshot(db, 'ROMANTIC WARRIOR')) == before


@pytest.mark.parametrize('venue,number', [('Overseas', 7), (None, None), (None, 8)])
def test_name_match_alone_does_not_establish_local_identity(db, venue, number):
    insert(db, 'ROMANTIC WARRIOR', venue, number)
    before = tuple(snapshot(db, 'ROMANTIC WARRIOR'))
    db.backfill_card_fields('2024/06/02', card([
        {'name': 'ROMANTIC WARRIOR', 'number': 7, 'rating': 40, 'gear': 'B'}]))
    assert tuple(snapshot(db, 'ROMANTIC WARRIOR')) == before


def test_matching_runner_number_can_repair_missing_local_venue(db):
    insert(db, 'HAPPY HERO', None, 7)
    db.backfill_card_fields('2024/06/02', card([
        {'name': 'HAPPY HERO', 'number': 7, 'rating': 40, 'gear': 'B'}]))
    row = snapshot(db, 'HAPPY HERO')
    assert (row['venue'], row['distance'], row['class_type'], row['rating'], row['gear']) == (
        'Sha Tin', 2000, 'Class 5', 40, 'B')


def test_existing_local_runner_keeps_manual_horse_fields(db):
    insert(db, 'HAPPY HERO', 'Sha Tin', 7)
    db.conn.execute("UPDATE race_results SET rating=45,gear='TT' WHERE horse_name='HAPPY HERO'")
    db.backfill_card_fields('2024/06/02', card([
        {'name': 'HAPPY HERO', 'number': 7, 'rating': 40, 'gear': 'B'}]))
    row = snapshot(db, 'HAPPY HERO')
    assert (row['distance'], row['rating'], row['gear']) == (2000, 45, 'TT')


def test_happy_valley_legacy_card_without_runner_number(db):
    insert(db, 'HAPPY HERO', 'Happy Valley', 7)
    db.backfill_card_fields('2024/06/02', card([
        {'name': 'HAPPY HERO', 'rating': 40}], venue='Happy Valley'))
    row = snapshot(db, 'HAPPY HERO')
    assert (row['venue'], row['rating']) == ('Happy Valley', 40)


def test_conflicting_local_runner_number_is_untouched(db):
    insert(db, 'HAPPY HERO', 'Sha Tin', 8)
    before = tuple(snapshot(db, 'HAPPY HERO'))
    db.backfill_card_fields('2024/06/02', card([
        {'name': 'HAPPY HERO', 'number': 7, 'rating': 40}]))
    assert tuple(snapshot(db, 'HAPPY HERO')) == before


def test_foreign_card_cannot_relabel_local_runner(db):
    insert(db, 'HAPPY HERO', 'Sha Tin', 7)
    before = tuple(snapshot(db, 'HAPPY HERO'))
    db.backfill_card_fields('2024/06/02', card([
        {'name': 'HAPPY HERO', 'number': 7, 'rating': 40}], venue='Tokyo'))
    assert tuple(snapshot(db, 'HAPPY HERO')) == before

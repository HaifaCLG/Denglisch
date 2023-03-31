import argparse
import datetime
import json
import os
import requests
import time
import urllib


# default start and end times
DEFAULT_START_TIME = '2019-01-01'
DEFAULT_END_TIME = '2019-02-01'

# timeout for individual requests in seconds
TIMEOUT_REQ = 60

# number of consecutive failed requests after which to abort
MAX_FAILED_REQ = 10

# directory to store downloaded comments
OUT_DIR = '/Users/do/Documents/GermanEnglish/results'

PUSHSHIFT_ENDPOINT = 'https://api.pushshift.io/reddit/search/comment'


def build_base_query(subreddit=None, author=None):
    params = []
    if subreddit:
        subreddit_quoted = urllib.parse.quote(subreddit, safe='')
        params.append(f'subreddit={subreddit_quoted}')
    if author:
        author_quoted = urllib.parse.quote(author, safe='')
        params.append(f'author={author_quoted}')
    return PUSHSHIFT_ENDPOINT + '?' + '&'.join(params)


def build_task_name(subreddit=None, author=None):
    task_components = []
    if subreddit:
        task_components.append(f'subreddit "{subreddit}"')
    if author:
        task_components.append(f'author "{author}"')
    return ' and '.join(task_components)


def fail_count_to_sleep_time(c):
    return c**2


def send_query(query):
    # try query and catch any exceptions but KeyboardInterrupt and SystemExit
    try:
        return requests.get(query, timeout=TIMEOUT_REQ)
    except Exception:
        return None


def has_text(element):
    body = element.get('body', '')
    selftext = element.get('selftext', '')
    return len(body) > 0 and body != '[removed]' or len(selftext) > 0 and selftext != '[removed]'


def download_comments(start_timestamp, end_timestamp, out_file_name, resume_file_name, subreddit=None, author=None):
    if not subreddit and not author:
        return

    task = build_task_name(subreddit=subreddit, author=author)
    base_query = build_base_query(subreddit=subreddit, author=author)

    print(f'crawling: {task}\n')

    # try to resume from where we left off
    try:
        with open(resume_file_name) as resume_file:
            cur_timestamp = int(resume_file.read())
            assert cur_timestamp >= 0
            print('resuming from timestamp: {cur_timestamp}')
    except:
        cur_timestamp = end_timestamp

    # crawl posts from subreddit/author
    req_count = 0
    fail_count = 0
    with open(out_file_name, 'a') as out_file:
        while fail_count < MAX_FAILED_REQ:
            # wait a few seconds if last request failed
            if fail_count:
                sleep_time = fail_count_to_sleep_time(fail_count)
                print(f'retrying in: {sleep_time} s\n')
                time.sleep(sleep_time)

            req_count += 1

            # build query
            params = ['sort=created_utc', 'size=100', 'before=' + str(cur_timestamp), 'after=' + str(start_timestamp)]
            query = base_query + ('' if base_query.endswith('?') else '&') + '&'.join(params)

            print(f'REQUEST #{req_count}')
            print(f'current timestamp: {cur_timestamp}')
            print(f'request: {query}')

            r = send_query(query)
            if not r:
                print('request failed\n')
                fail_count += 1
                continue

            if r.status_code != 200:
                print(f'bad response code: {r.status_code}\n')
                fail_count += 1
                continue
            else:
                fail_count = 0

            response_data = r.json()['data']

            print(f'retrieved items: {len(response_data)}\n')

            if len(response_data) == 0:
                print(f'finished: {task}\n')
                break

            # record the response
            for element in response_data:
                if has_text(element):
                    json.dump(element, out_file)
                    out_file.write('\n')
                    out_file.flush()

                cur_timestamp = element['created_utc']

    # if aborted, store current timestamp
    if fail_count:
        print('=====================================')
        print('ABORTING: TOO MANY FAILED REQUESTS!!!')
        print('=====================================')
        print('restart the script to resume from this point')
        with open(resume_file_name, 'w') as resume_file:
            resume_file.write(str(cur_timestamp) + '\n')


def str_to_timestamp(s, tzinfo):
    ymd_str = s.split('-')
    if len(ymd_str) != 3:
        raise ValueError

    ymd_int = map(int, ymd_str)
    dtime = datetime.datetime(*ymd_int, tzinfo=tzinfo)
    return int(dtime.timestamp())


if __name__ == '__main__':
    def date(s):
        return s, str_to_timestamp(s, datetime.timezone.utc)

    def comma_separated(s):
        return s.split(',')

    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-s', '--start-time',
        metavar='YYYY-MM-DD',
        type=date,
        default=date(DEFAULT_START_TIME),
        help='default: ' + DEFAULT_START_TIME
    )
    ap.add_argument(
        '-e', '--end-time',
        metavar='YYYY-MM-DD',
        type=date,
        default=date(DEFAULT_END_TIME),
        help='default: ' + DEFAULT_END_TIME
    )
    ap.add_argument(
        '-r', '--subreddits',
        metavar='SUBREDDIT,...',
        type=comma_separated,
        default=[],
        help='download comments posted in these subreddits'
    )
    ap.add_argument(
        '-a', '--authors',
        metavar='USERNAME,...',
        type=comma_separated,
        default=[],
        help='download comments posted by these authors'
    )
    args = ap.parse_args()

    start_time = args.start_time[0]
    start_timestamp = args.start_time[1]

    end_time = args.end_time[0]
    end_timestamp = args.end_time[1]

    if not args.subreddits and not args.authors:
        print('please supply at least one subreddit or author\n')
        ap.print_help()
        exit(1)

    for subreddit in args.subreddits:
        resume_file = os.path.join(OUT_DIR, f'.subreddit.{subreddit}.resume')
        out_file = os.path.join(OUT_DIR, f'subreddit.{subreddit}.{start_time}--{end_time}.comment.json.out')
        download_comments(start_timestamp, end_timestamp, out_file, resume_file, subreddit=subreddit)

    for author in args.authors:
        resume_file = os.path.join(OUT_DIR, f'.author.{author}.resume')
        out_file = os.path.join(OUT_DIR, f'author.{author}.{start_time}--{end_time}.comment.json.out')
        download_comments(start_timestamp, end_timestamp, out_file, resume_file, author=author)

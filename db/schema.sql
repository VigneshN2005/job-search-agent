create extension if not exists vector;

create table if not exists jobs (
    id              bigserial primary key,
    source          text        not null,
    source_id       text        not null,
    title           text        not null,
    company         text        not null,
    location        text        not null default '',
    remote          boolean     not null default false,
    job_type        text        not null default '',
    salary_min      integer,
    salary_max      integer,
    salary_currency text,
    tags            text[]      not null default '{}',
    description     text        not null default '',
    url             text        not null,
    posted_at       timestamptz,
    embedding       vector(768),
    created_at      timestamptz not null default now(),
    unique (source, source_id)
);

create index if not exists jobs_embedding_hnsw on jobs using hnsw (embedding vector_cosine_ops);
create index if not exists jobs_posted_at_idx on jobs (posted_at desc);
create index if not exists jobs_remote_idx on jobs (remote);
create index if not exists jobs_tags_gin on jobs using gin (tags);

create table if not exists sessions (
    id          uuid primary key,
    created_at  timestamptz not null default now()
);

create table if not exists messages (
    id          bigserial primary key,
    session_id  uuid        not null references sessions(id) on delete cascade,
    role        text        not null check (role in ('user', 'assistant')),
    content     text        not null,
    created_at  timestamptz not null default now()
);
create index if not exists messages_session_idx on messages (session_id, id);

create table if not exists preferences (
    session_id  uuid        not null references sessions(id) on delete cascade,
    key         text        not null,
    value       text        not null,
    updated_at  timestamptz not null default now(),
    primary key (session_id, key)
);

alter table jobs enable row level security;
alter table sessions enable row level security;
alter table messages enable row level security;
alter table preferences enable row level security;

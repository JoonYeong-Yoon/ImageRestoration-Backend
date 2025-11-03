from db.connection import engine, Base

# ensure model modules are imported so their metadata is registered
import db_models.user_model  # noqa: F401
import db_models.session_model  # noqa: F401

# Create all tables
def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("Database tables created successfully!")
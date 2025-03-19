"""MCP Server Package."""

from . import server

__version__ = "0.1.0"
__all__ = ["main", "server"]


def main():
    # get the main from server
    server.main()


if __name__ == "__main__":
    main()

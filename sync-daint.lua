settings {
    nodaemon = true,
    statusInterval = 1
}

sync {
    default.rsyncssh,
    source = "/home/gag/Libraries/qtpyt/",
    targetdir = "/users/ggandus/Libraries/qtpyt/",
    delay = 1, 
    host = "daint",
    rsync = {
        --include = {"*.py"},
        --exclude = {"/*"}
    }
}

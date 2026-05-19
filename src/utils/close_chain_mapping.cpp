// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2026 Luo1imasi

#include "close_chain_mapping.hpp"
#include "decouple_atom01.hpp"

std::shared_ptr<Decouple> Decouple::create(const std::string &type)
{
    if (type == "atom01")
    {
        return std::make_shared<DecoupleAtom01>();
    }
    else
    {
        throw std::runtime_error("Unknown close_chain type: " + type +
                                 ". Supported types: atom01");
    }
}
